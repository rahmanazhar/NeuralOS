/// @file vmm.cpp
/// @brief Virtual Memory Manager implementation.
///
/// Loads experts from .nxp files via PlatformIO, caches them in slab-allocated
/// buffers, and manages eviction via CLOCK-Pro. Per-page atomic state tracks
/// EVICTED/LOADING/RESIDENT/CACHED transitions. Opaque handles with generation
/// counters detect stale references after eviction+reload cycles.
///
/// Budget-aware factory (Vmm::create) reads .nxp header, computes budget
/// partition, validates sufficiency, pre-allocates KV cache, and tracks
/// runtime statistics (hit rate, eviction count, etc.).

#include "vmm/vmm.h"
#include "vmm/memory_budget.h"
#include "vmm/clock_pro.h"
#include "vmm/slab_allocator.h"

#include "format/expert_format.h"
#include "format/crc32.h"
#include "io/platform_io.h"
#include "io/async_io_pool.h"

#include <atomic>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace nos {

// ── Internal page entry ─────────────────────────────────────────────────────

struct PageEntry {
    std::atomic<uint8_t>  state{static_cast<uint8_t>(PageState::EVICTED)};
    std::atomic<uint32_t> generation{0};
    std::atomic<uint32_t> refcount{0};
    uint8_t*  data{nullptr};       // Pointer into slab buffer
    int       slab_slot{-1};       // Slab slot index (-1 = no slot)
    uint64_t  weight_offset{0};    // File offset for weight data
    uint64_t  weight_size{0};      // Size of weight data in bytes
    uint64_t  scale_offset{0};     // File offset for scale data
    uint32_t  scale_size{0};       // Size of scale data in bytes
    uint32_t  crc32_expected{0};   // Expected CRC32C of weight data
};

// ── VmmImpl (PIMPL) ─────────────────────────────────────────────────────────

class VmmImpl {
public:
    explicit VmmImpl(VmmConfig config);
    ~VmmImpl();

    ExpertHandle get_handle(uint32_t layer_id, uint32_t expert_id) const;
    const uint8_t* pin(ExpertHandle handle);
    void unpin(ExpertHandle handle);
    PageState page_state(ExpertHandle handle) const;
    size_t resident_count() const;
    size_t cached_count() const;
    size_t load_count() const;

    // Budget-aware extensions
    VmmStats stats() const;
    const BudgetPartition& budget() const { return budget_; }
    void set_budget(const BudgetPartition& bp) { budget_ = bp; }

    void* kv_cache_base() const { return kv_cache_; }
    size_t kv_cache_size() const { return kv_cache_bytes_; }
    void allocate_kv_cache(size_t bytes);

    // Async I/O extension (Phase 4)
    bool pin_async(ExpertHandle handle);
    const uint8_t* await_pin(ExpertHandle handle);
    void prefetch_expert(uint32_t layer_id, uint32_t expert_id);

private:
    void load_expert(uint32_t page_index);
    void load_expert_async(uint32_t page_index);
    void do_evict(uint32_t page_index);

    VmmConfig config_;

    // Page table: allocated as a unique_ptr to a raw array because PageEntry
    // contains std::atomic members (non-copyable, non-movable), which prevents
    // std::vector::resize from working.
    std::unique_ptr<PageEntry[]> page_table_;

    std::unique_ptr<SlabAllocator> slab_;
    std::unique_ptr<ClockPro> clock_pro_;
    std::unique_ptr<PlatformIO> io_;
    int nxp_fd_{-1};

    // Pre-loaded expert entry metadata from NxpReader
    std::vector<NxpExpertEntry> entries_;

    uint32_t total_experts_{0};
    size_t slot_count_{0};

    // Load counter for testing (backward compat)
    std::atomic<size_t> load_count_{0};

    // ── Runtime statistics ──────────────────────────────────────────────
    std::atomic<uint64_t> stat_total_pins_{0};
    std::atomic<uint64_t> stat_cache_hits_{0};
    std::atomic<uint64_t> stat_cache_misses_{0};
    std::atomic<uint64_t> stat_evictions_{0};
    std::atomic<uint64_t> stat_crc_failures_{0};

    // ── Async I/O pool ────────────────────────────────────────────────────
    std::unique_ptr<AsyncIOPool> async_io_;

    // ── Budget and KV cache ─────────────────────────────────────────────
    BudgetPartition budget_{};
    void* kv_cache_{nullptr};
    size_t kv_cache_bytes_{0};
};

VmmImpl::VmmImpl(VmmConfig config) : config_(std::move(config)) {
    // Step 1: Load expert index from NxpReader
    NxpReader reader;
    if (!reader.open(config_.nxp_path)) {
        assert(false && "Failed to open .nxp file");
        return;
    }

    const auto& hdr = reader.header();
    total_experts_ = hdr.num_layers * hdr.experts_per_layer;

    // Pre-load all expert entries
    entries_.resize(total_experts_);
    for (uint32_t l = 0; l < hdr.num_layers; ++l) {
        for (uint32_t e = 0; e < hdr.experts_per_layer; ++e) {
            const NxpExpertEntry* entry = reader.find_expert(l, e);
            if (entry) {
                uint32_t dense_id = l * hdr.experts_per_layer + e;
                entries_[dense_id] = *entry;
            }
        }
    }
    reader.close();

    // Step 2: Compute max expert size if not provided
    if (config_.max_expert_size == 0) {
        for (const auto& ent : entries_) {
            size_t total_size = ent.size + ent.scale_size;
            if (total_size > config_.max_expert_size) {
                config_.max_expert_size = total_size;
            }
        }
        // Align up to 64
        config_.max_expert_size =
            (config_.max_expert_size + 63) & ~static_cast<size_t>(63);
    }

    // Step 3: Open fd for PlatformIO reads
    nxp_fd_ = ::open(config_.nxp_path.c_str(), O_RDONLY);
    assert(nxp_fd_ >= 0 && "Failed to open .nxp file for I/O");

    // Step 4: Create PlatformIO backend
    io_ = PlatformIO::create();

    // Step 4b: Create AsyncIOPool (2 I/O threads on macOS, delegates to io_uring on Linux)
    async_io_ = std::make_unique<AsyncIOPool>(2, io_.get());

    // Step 5: Create slab allocator
    slot_count_ = config_.expert_cache_bytes / config_.max_expert_size;
    if (slot_count_ == 0) slot_count_ = 1;
    slab_ = std::make_unique<SlabAllocator>(slot_count_, config_.max_expert_size);

    // Step 6: Create CLOCK-Pro
    clock_pro_ = std::make_unique<ClockPro>(slot_count_, slot_count_ * 2);

    // Step 7: Initialize page table (all EVICTED)
    page_table_ = std::make_unique<PageEntry[]>(total_experts_);
    for (uint32_t i = 0; i < total_experts_; ++i) {
        page_table_[i].weight_offset = entries_[i].offset;
        page_table_[i].weight_size = entries_[i].size;
        page_table_[i].scale_offset = entries_[i].scale_offset;
        page_table_[i].scale_size = entries_[i].scale_size;
        page_table_[i].crc32_expected = entries_[i].crc32;
    }
}

VmmImpl::~VmmImpl() {
    if (nxp_fd_ >= 0) {
        ::close(nxp_fd_);
        nxp_fd_ = -1;
    }
    if (kv_cache_) {
        ::free(kv_cache_);
        kv_cache_ = nullptr;
    }
}

void VmmImpl::allocate_kv_cache(size_t bytes) {
    if (bytes == 0) return;
    // 64-byte aligned contiguous allocation
    kv_cache_bytes_ = bytes;
#if defined(__APPLE__) || defined(_WIN32)
    // posix_memalign or aligned allocation
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, 64, bytes) != 0) {
        ptr = nullptr;
    }
    kv_cache_ = ptr;
#else
    kv_cache_ = std::aligned_alloc(64, bytes);
#endif
    if (kv_cache_) {
        std::memset(kv_cache_, 0, bytes);
    }
}

ExpertHandle VmmImpl::get_handle(uint32_t layer_id, uint32_t expert_id) const {
    uint32_t index = layer_id * config_.experts_per_layer + expert_id;
    if (index >= total_experts_) {
        return INVALID_HANDLE;
    }
    ExpertHandle h{};
    h.index = index;
    h.generation = page_table_[index].generation.load(std::memory_order_acquire);
    return h;
}

const uint8_t* VmmImpl::pin(ExpertHandle handle) {
    if (handle.index >= total_experts_) return nullptr;

    PageEntry& page = page_table_[handle.index];

    // Validate generation -- stale handle returns nullptr
    if (handle.generation != page.generation.load(std::memory_order_acquire)) {
        return nullptr;
    }

    // Track total pins
    stat_total_pins_.fetch_add(1, std::memory_order_relaxed);

    // Check current state
    auto state = static_cast<PageState>(page.state.load(std::memory_order_acquire));

    // Spin-wait if LOADING or PREFETCH
    while (state == PageState::LOADING || state == PageState::PREFETCH) {
        state = static_cast<PageState>(page.state.load(std::memory_order_acquire));
    }

    if (state == PageState::EVICTED) {
        // Cache miss: load from NVMe
        stat_cache_misses_.fetch_add(1, std::memory_order_relaxed);
        load_expert(handle.index);
        state = static_cast<PageState>(page.state.load(std::memory_order_acquire));
    } else if (state == PageState::RESIDENT || state == PageState::CACHED) {
        // Cache hit: already in RAM
        stat_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    }

    if (state == PageState::RESIDENT || state == PageState::CACHED) {
        // Re-check generation after potential load (generation increments on evict+reload)
        if (handle.generation != page.generation.load(std::memory_order_acquire)) {
            return nullptr;
        }

        // Pin: increment refcount, transition to CACHED
        page.refcount.fetch_add(1, std::memory_order_acq_rel);
        page.state.store(static_cast<uint8_t>(PageState::CACHED),
                         std::memory_order_release);

        // Mark as referenced for CLOCK-Pro
        clock_pro_->mark_accessed(handle.index);

        return page.data;
    }

    return nullptr;  // Load failed (CRC mismatch or other error)
}

void VmmImpl::unpin(ExpertHandle handle) {
    if (handle.index >= total_experts_) return;

    PageEntry& page = page_table_[handle.index];

    uint32_t prev = page.refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        // Last unpin: transition CACHED -> RESIDENT (now evictable)
        page.state.store(static_cast<uint8_t>(PageState::RESIDENT),
                         std::memory_order_release);
    }
}

PageState VmmImpl::page_state(ExpertHandle handle) const {
    if (handle.index >= total_experts_) return PageState::EVICTED;
    return static_cast<PageState>(
        page_table_[handle.index].state.load(std::memory_order_acquire));
}

size_t VmmImpl::resident_count() const {
    size_t count = 0;
    for (uint32_t i = 0; i < total_experts_; ++i) {
        auto s = static_cast<PageState>(
            page_table_[i].state.load(std::memory_order_relaxed));
        if (s == PageState::RESIDENT || s == PageState::CACHED) {
            ++count;
        }
    }
    return count;
}

size_t VmmImpl::cached_count() const {
    size_t count = 0;
    for (uint32_t i = 0; i < total_experts_; ++i) {
        auto s = static_cast<PageState>(
            page_table_[i].state.load(std::memory_order_relaxed));
        if (s == PageState::CACHED) {
            ++count;
        }
    }
    return count;
}

size_t VmmImpl::load_count() const {
    return load_count_.load(std::memory_order_relaxed);
}

VmmStats VmmImpl::stats() const {
    VmmStats s{};
    s.total_pins    = stat_total_pins_.load(std::memory_order_relaxed);
    s.cache_hits    = stat_cache_hits_.load(std::memory_order_relaxed);
    s.cache_misses  = stat_cache_misses_.load(std::memory_order_relaxed);
    s.evictions     = stat_evictions_.load(std::memory_order_relaxed);
    s.crc_failures  = stat_crc_failures_.load(std::memory_order_relaxed);

    // Compute current page counts
    uint32_t resident = 0, cached = 0, evicted = 0;
    for (uint32_t i = 0; i < total_experts_; ++i) {
        auto st = static_cast<PageState>(
            page_table_[i].state.load(std::memory_order_relaxed));
        switch (st) {
            case PageState::RESIDENT: ++resident; break;
            case PageState::CACHED:   ++cached; break;
            case PageState::EVICTED:  ++evicted; break;
            default: break;
        }
    }
    s.resident_pages = resident;
    s.cached_pages   = cached;
    s.evicted_pages  = evicted;
    s.hit_rate = (s.total_pins > 0)
                 ? static_cast<double>(s.cache_hits) / static_cast<double>(s.total_pins)
                 : 0.0;
    return s;
}

void VmmImpl::load_expert(uint32_t page_index) {
    PageEntry& page = page_table_[page_index];

    // CAS transition EVICTED -> LOADING
    uint8_t expected = static_cast<uint8_t>(PageState::EVICTED);
    if (!page.state.compare_exchange_strong(
            expected, static_cast<uint8_t>(PageState::LOADING),
            std::memory_order_acq_rel)) {
        return;  // Another thread is already loading
    }

    // Allocate slab buffer
    auto [buf, slot] = slab_->allocate();
    if (!buf) {
        // Slab full: evict to make space
        for (size_t attempt = 0; attempt < slot_count_; ++attempt) {
            uint32_t victim_idx = clock_pro_->evict_one();
            if (victim_idx == UINT32_MAX) break;

            // Check that the eviction candidate is actually evictable
            PageEntry& victim = page_table_[victim_idx];
            auto victim_state = static_cast<PageState>(
                victim.state.load(std::memory_order_acquire));

            if (victim_state == PageState::RESIDENT &&
                victim.refcount.load(std::memory_order_acquire) == 0) {
                do_evict(victim_idx);
                std::tie(buf, slot) = slab_->allocate();
                if (buf) break;
            }
        }

        if (!buf) {
            // Cannot allocate even after eviction attempts
            page.state.store(static_cast<uint8_t>(PageState::EVICTED),
                             std::memory_order_release);
            return;
        }
    }

    // Read weight data via PlatformIO
    io_->submit_read(nxp_fd_, buf,
                     static_cast<size_t>(page.weight_size),
                     static_cast<off_t>(page.weight_offset), nullptr);

    IoCompletion comp{};
    while (io_->poll(&comp, 1, 100) == 0) {}

    if (comp.result < 0 ||
        static_cast<uint64_t>(comp.result) != page.weight_size) {
        // Read failed
        slab_->free(slot);
        page.state.store(static_cast<uint8_t>(PageState::EVICTED),
                         std::memory_order_release);
        return;
    }

    // Read scale data (stored after aligned weights in the same slab slot)
    if (page.scale_size > 0) {
        size_t aligned_weight =
            (static_cast<size_t>(page.weight_size) + 63) & ~size_t(63);
        io_->submit_read(nxp_fd_, buf + aligned_weight,
                         static_cast<size_t>(page.scale_size),
                         static_cast<off_t>(page.scale_offset), nullptr);

        IoCompletion comp2{};
        while (io_->poll(&comp2, 1, 100) == 0) {}
    }

    // Verify CRC32C of weight data
    uint32_t computed_crc = crc32c(buf, static_cast<size_t>(page.weight_size));
    if (computed_crc != page.crc32_expected) {
        // CRC mismatch -- data corruption
        stat_crc_failures_.fetch_add(1, std::memory_order_relaxed);
        slab_->free(slot);
        page.state.store(static_cast<uint8_t>(PageState::EVICTED),
                         std::memory_order_release);
        return;
    }

    // Success: update page entry
    page.data = buf;
    page.slab_slot = slot;
    page.state.store(static_cast<uint8_t>(PageState::RESIDENT),
                     std::memory_order_release);

    // Insert into CLOCK-Pro
    clock_pro_->insert(page_index);

    // Increment load counter
    load_count_.fetch_add(1, std::memory_order_relaxed);
}

void VmmImpl::do_evict(uint32_t page_index) {
    PageEntry& page = page_table_[page_index];

    // Free slab slot
    if (page.slab_slot >= 0) {
        slab_->free(page.slab_slot);
    }

    page.data = nullptr;
    page.slab_slot = -1;

    // Increment generation (stale handles will fail validation)
    page.generation.fetch_add(1, std::memory_order_acq_rel);

    // Transition to EVICTED
    page.state.store(static_cast<uint8_t>(PageState::EVICTED),
                     std::memory_order_release);

    // Track eviction
    stat_evictions_.fetch_add(1, std::memory_order_relaxed);
}

// ── Async I/O extension ──────────────────────────────────────────────────────

void VmmImpl::load_expert_async(uint32_t page_index) {
    PageEntry& page = page_table_[page_index];

    // CAS transition EVICTED -> LOADING (only one thread wins)
    uint8_t expected = static_cast<uint8_t>(PageState::EVICTED);
    if (!page.state.compare_exchange_strong(
            expected, static_cast<uint8_t>(PageState::LOADING),
            std::memory_order_acq_rel)) {
        return;  // Another thread is already loading, or page is already loaded
    }

    // Allocate slab buffer
    auto [buf, slot] = slab_->allocate();
    if (!buf) {
        // Slab full: evict to make space
        for (size_t attempt = 0; attempt < slot_count_; ++attempt) {
            uint32_t victim_idx = clock_pro_->evict_one();
            if (victim_idx == UINT32_MAX) break;

            PageEntry& victim = page_table_[victim_idx];
            auto victim_state = static_cast<PageState>(
                victim.state.load(std::memory_order_acquire));

            if (victim_state == PageState::RESIDENT &&
                victim.refcount.load(std::memory_order_acquire) == 0) {
                do_evict(victim_idx);
                std::tie(buf, slot) = slab_->allocate();
                if (buf) break;
            }
        }

        if (!buf) {
            page.state.store(static_cast<uint8_t>(PageState::EVICTED),
                             std::memory_order_release);
            return;
        }
    }

    // Store slab info so await_pin can find the data pointer
    page.data = buf;
    page.slab_slot = slot;

    // Submit async read for weight data
    async_io_->submit_async_read(
        nxp_fd_, buf,
        static_cast<size_t>(page.weight_size),
        static_cast<off_t>(page.weight_offset),
        [this, page_index, buf, slot](int result) {
            PageEntry& pg = page_table_[page_index];

            if (result < 0 ||
                static_cast<uint64_t>(result) != pg.weight_size) {
                // Read failed
                slab_->free(slot);
                pg.data = nullptr;
                pg.slab_slot = -1;
                pg.state.store(static_cast<uint8_t>(PageState::EVICTED),
                               std::memory_order_release);
                return;
            }

            // Read scale data synchronously (small, fast)
            if (pg.scale_size > 0) {
                size_t aligned_weight =
                    (static_cast<size_t>(pg.weight_size) + 63) & ~size_t(63);
                ssize_t sb = ::pread(nxp_fd_, buf + aligned_weight,
                                     static_cast<size_t>(pg.scale_size),
                                     static_cast<off_t>(pg.scale_offset));
                (void)sb;  // Scale read failure is non-fatal
            }

            // Verify CRC32C
            uint32_t computed_crc = crc32c(buf, static_cast<size_t>(pg.weight_size));
            if (computed_crc != pg.crc32_expected) {
                stat_crc_failures_.fetch_add(1, std::memory_order_relaxed);
                slab_->free(slot);
                pg.data = nullptr;
                pg.slab_slot = -1;
                pg.state.store(static_cast<uint8_t>(PageState::EVICTED),
                               std::memory_order_release);
                return;
            }

            // Increment load counter before state transition so that
            // any thread observing RESIDENT via acquire also sees the counter
            load_count_.fetch_add(1, std::memory_order_relaxed);

            // Insert into CLOCK-Pro
            clock_pro_->insert(page_index);

            // Transition LOADING -> RESIDENT (release makes above visible)
            pg.state.store(static_cast<uint8_t>(PageState::RESIDENT),
                           std::memory_order_release);
        });
}

bool VmmImpl::pin_async(ExpertHandle handle) {
    if (handle.index >= total_experts_) return false;

    PageEntry& page = page_table_[handle.index];

    // Check current state
    auto state = static_cast<PageState>(page.state.load(std::memory_order_acquire));

    // Already loaded or loading: idempotent success
    if (state == PageState::LOADING || state == PageState::RESIDENT ||
        state == PageState::CACHED || state == PageState::PREFETCH) {
        return true;
    }

    // EVICTED: initiate async load
    if (state == PageState::EVICTED) {
        load_expert_async(handle.index);
        return true;
    }

    return false;
}

const uint8_t* VmmImpl::await_pin(ExpertHandle handle) {
    if (handle.index >= total_experts_) return nullptr;

    PageEntry& page = page_table_[handle.index];

    // Validate generation
    if (handle.generation != page.generation.load(std::memory_order_acquire)) {
        return nullptr;
    }

    // Track total pins
    stat_total_pins_.fetch_add(1, std::memory_order_relaxed);

    // Spin-wait while LOADING, draining completions to progress I/O
    auto state = static_cast<PageState>(page.state.load(std::memory_order_acquire));
    while (state == PageState::LOADING || state == PageState::PREFETCH) {
        async_io_->drain_completions();
        state = static_cast<PageState>(page.state.load(std::memory_order_acquire));
    }

    if (state == PageState::EVICTED) {
        // Not yet loaded and not loading -- fallback to synchronous load
        stat_cache_misses_.fetch_add(1, std::memory_order_relaxed);
        load_expert(handle.index);
        state = static_cast<PageState>(page.state.load(std::memory_order_acquire));
    } else if (state == PageState::RESIDENT || state == PageState::CACHED) {
        stat_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    }

    if (state == PageState::RESIDENT || state == PageState::CACHED) {
        // Re-check generation after potential load
        if (handle.generation != page.generation.load(std::memory_order_acquire)) {
            return nullptr;
        }

        // Pin: increment refcount, transition to CACHED
        page.refcount.fetch_add(1, std::memory_order_acq_rel);
        page.state.store(static_cast<uint8_t>(PageState::CACHED),
                         std::memory_order_release);

        clock_pro_->mark_accessed(handle.index);
        return page.data;
    }

    return nullptr;
}

void VmmImpl::prefetch_expert(uint32_t layer_id, uint32_t expert_id) {
    ExpertHandle handle = get_handle(layer_id, expert_id);
    if (handle == INVALID_HANDLE) return;
    pin_async(handle);
}

// ── Vmm public interface (delegates to VmmImpl) ─────────────────────────────

Vmm::Vmm(VmmConfig config) : impl_(new VmmImpl(std::move(config))) {}
Vmm::~Vmm() { delete impl_; }

Vmm::Vmm(Vmm&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

Vmm& Vmm::operator=(Vmm&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

ExpertHandle Vmm::get_handle(uint32_t layer_id, uint32_t expert_id) const {
    return impl_->get_handle(layer_id, expert_id);
}

const uint8_t* Vmm::pin(ExpertHandle handle) {
    return impl_->pin(handle);
}

void Vmm::unpin(ExpertHandle handle) {
    impl_->unpin(handle);
}

PageState Vmm::page_state(ExpertHandle handle) const {
    return impl_->page_state(handle);
}

size_t Vmm::resident_count() const {
    return impl_->resident_count();
}

size_t Vmm::cached_count() const {
    return impl_->cached_count();
}

size_t Vmm::load_count() const {
    return impl_->load_count();
}

VmmStats Vmm::stats() const {
    return impl_->stats();
}

const BudgetPartition& Vmm::budget() const {
    return impl_->budget();
}

void* Vmm::kv_cache_base() const {
    return impl_->kv_cache_base();
}

size_t Vmm::kv_cache_size() const {
    return impl_->kv_cache_size();
}

bool Vmm::pin_async(ExpertHandle handle) {
    return impl_->pin_async(handle);
}

const uint8_t* Vmm::await_pin(ExpertHandle handle) {
    return impl_->await_pin(handle);
}

void Vmm::prefetch_expert(uint32_t layer_id, uint32_t expert_id) {
    impl_->prefetch_expert(layer_id, expert_id);
}

// ── Vmm::create() factory ───────────────────────────────────────────────────

std::unique_ptr<Vmm> Vmm::create(const VmmFullConfig& config) {
    // Step 1: Open .nxp via NxpReader to get header
    NxpReader reader;
    if (!reader.open(config.nxp_path)) {
        std::fprintf(stderr, "Error: Failed to open .nxp file: %s\n",
                     config.nxp_path.c_str());
        return nullptr;
    }

    const auto& hdr = reader.header();

    // Step 2: Find max_expert_size by scanning entries
    size_t max_expert_size = 0;
    for (uint32_t l = 0; l < hdr.num_layers; ++l) {
        for (uint32_t e = 0; e < hdr.experts_per_layer; ++e) {
            const NxpExpertEntry* entry = reader.find_expert(l, e);
            if (entry) {
                size_t entry_size = entry->size + entry->scale_size;
                if (entry_size > max_expert_size) {
                    max_expert_size = entry_size;
                }
            }
        }
    }
    // Align up to 64
    max_expert_size = (max_expert_size + 63) & ~size_t(63);
    reader.close();

    // Step 3: Construct ModelParams from header + config
    ModelParams params{};
    params.n_layers          = hdr.num_layers;
    params.n_kv_heads        = config.n_kv_heads;
    params.head_dim          = config.head_dim;
    params.hidden_dim        = hdr.hidden_dim;
    params.max_expert_size   = static_cast<uint32_t>(
        std::min(max_expert_size, static_cast<size_t>(UINT32_MAX)));
    params.top_k             = config.top_k;
    params.experts_per_layer = hdr.experts_per_layer;

    // Step 4: Compute budget
    BudgetPartition bp = compute_budget(
        config.user_budget_bytes, params, config.desired_context_length);

    // Step 5: Validate
    if (!bp.sufficient) {
        std::string report = format_budget_report(bp, params);
        std::fprintf(stderr, "%s", report.c_str());
        return nullptr;
    }

    // Step 6: Print budget report to stderr (always, for visibility)
    {
        std::string report = format_budget_report(bp, params);
        std::fprintf(stderr, "%s", report.c_str());
    }

    // Step 7: Construct VmmConfig from budget
    VmmConfig vmm_config{};
    vmm_config.expert_cache_bytes = bp.expert_cache;
    vmm_config.max_expert_size    = max_expert_size;
    vmm_config.num_layers         = hdr.num_layers;
    vmm_config.experts_per_layer  = hdr.experts_per_layer;
    vmm_config.nxp_path           = config.nxp_path;

    // Step 8: Create VMM and set budget
    auto vmm = std::make_unique<Vmm>(std::move(vmm_config));
    vmm->impl_->set_budget(bp);

    // Step 9: Allocate KV cache (contiguous, pre-reserved)
    vmm->impl_->allocate_kv_cache(bp.kv_cache);

    return vmm;
}

}  // namespace nos
