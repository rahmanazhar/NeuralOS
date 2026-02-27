/// @file vmm.cpp
/// @brief Virtual Memory Manager implementation.
///
/// Loads experts from .nxp files via PlatformIO, caches them in slab-allocated
/// buffers, and manages eviction via CLOCK-Pro. Per-page atomic state tracks
/// EVICTED/LOADING/RESIDENT/CACHED transitions. Opaque handles with generation
/// counters detect stale references after eviction+reload cycles.

#include "vmm/vmm.h"
#include "vmm/memory_budget.h"
#include "vmm/clock_pro.h"
#include "vmm/slab_allocator.h"

#include "format/expert_format.h"
#include "format/crc32.h"
#include "io/platform_io.h"

#include <atomic>
#include <cassert>
#include <cstdio>
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

private:
    void load_expert(uint32_t page_index);
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

    // Load counter for testing
    std::atomic<size_t> load_count_{0};
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

    // Check current state
    auto state = static_cast<PageState>(page.state.load(std::memory_order_acquire));

    // Spin-wait if LOADING or PREFETCH
    while (state == PageState::LOADING || state == PageState::PREFETCH) {
        state = static_cast<PageState>(page.state.load(std::memory_order_acquire));
    }

    if (state == PageState::EVICTED) {
        // Cache miss: load from NVMe
        load_expert(handle.index);
        state = static_cast<PageState>(page.state.load(std::memory_order_acquire));
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

// ── Budget-aware factory and new methods (stubs for RED phase) ───────────────

std::unique_ptr<Vmm> Vmm::create(const VmmFullConfig& /*config*/) {
    return nullptr;  // Stub -- tests should fail
}

VmmStats Vmm::stats() const {
    VmmStats s{};
    return s;  // Stub
}

const BudgetPartition& Vmm::budget() const {
    static BudgetPartition empty{};
    return empty;  // Stub
}

void* Vmm::kv_cache_base() const {
    return nullptr;  // Stub
}

size_t Vmm::kv_cache_size() const {
    return 0;  // Stub
}

}  // namespace nos
