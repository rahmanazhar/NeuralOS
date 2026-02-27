#pragma once

/// @file vmm.h
/// @brief Virtual Memory Manager for expert caching with page table,
///        opaque handles, pin/unpin, and CLOCK-Pro eviction.
///
/// The VMM mediates every expert access: the inference engine, async I/O,
/// and prefetcher all flow through pin/unpin. Experts are loaded from
/// .nxp files via PlatformIO, verified via CRC32C, and cached in a
/// slab-allocated buffer pool managed by CLOCK-Pro eviction.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nos {

// Forward declaration for budget
struct BudgetPartition;

/// Opaque handle to a cached expert. Contains a dense index into the
/// page table and a generation counter that detects stale references
/// after eviction+reload cycles.
struct ExpertHandle {
    uint32_t index;       ///< Slot index in page table (dense expert ID)
    uint32_t generation;  ///< Incremented on each evict+reload cycle
};

inline bool operator==(ExpertHandle a, ExpertHandle b) {
    return a.index == b.index && a.generation == b.generation;
}
inline bool operator!=(ExpertHandle a, ExpertHandle b) {
    return !(a == b);
}

/// Invalid handle constant.
constexpr ExpertHandle INVALID_HANDLE{UINT32_MAX, 0};

/// Page state for an expert in the VMM cache.
enum class PageState : uint8_t {
    EVICTED   = 0,  ///< On NVMe only, no RAM buffer
    LOADING   = 1,  ///< NVMe -> RAM transfer in progress
    PREFETCH  = 2,  ///< Speculatively loading (lower priority)
    RESIDENT  = 3,  ///< In RAM, unpinned (evictable)
    CACHED    = 4,  ///< In RAM, pinned (non-evictable)
};

/// Configuration for the VMM (low-level, used by direct Vmm constructor).
struct VmmConfig {
    size_t   expert_cache_bytes;    ///< Total bytes for expert slab pool
    size_t   max_expert_size;       ///< Size of largest expert (weights + scales)
    uint32_t num_layers;            ///< Number of model layers
    uint32_t experts_per_layer;     ///< Number of experts per layer
    std::string nxp_path;           ///< Path to .nxp file
};

/// Full configuration for budget-aware VMM creation via Vmm::create().
struct VmmFullConfig {
    std::string nxp_path;                   ///< Path to .nxp file
    size_t      user_budget_bytes = 0;      ///< User --memory budget in bytes
    uint32_t    desired_context_length = 2048; ///< Desired KV context length
    uint32_t    n_kv_heads = 8;             ///< KV attention heads (default, set by inference engine)
    uint32_t    head_dim = 128;             ///< Dimension per head (default)
    uint32_t    top_k = 2;                  ///< Experts activated per token (default)
};

/// Runtime statistics snapshot from the VMM.
struct VmmStats {
    uint64_t total_pins;        ///< Total pin() calls
    uint64_t cache_hits;        ///< pin() calls that found RESIDENT/CACHED (no NVMe load)
    uint64_t cache_misses;      ///< pin() calls that triggered NVMe load
    uint64_t evictions;         ///< Total evictions performed
    uint64_t crc_failures;      ///< CRC32C verification failures
    uint32_t resident_pages;    ///< Current RESIDENT page count
    uint32_t cached_pages;      ///< Current CACHED (pinned) page count
    uint32_t evicted_pages;     ///< Current EVICTED page count
    double   hit_rate;          ///< cache_hits / total_pins (0.0 if no pins)
};

// Forward declarations for PIMPL
class VmmImpl;

/// Virtual Memory Manager for expert caching.
///
/// Loads experts from .nxp files via PlatformIO, caches them in slab-allocated
/// buffers, and manages eviction via CLOCK-Pro. All expert access goes through
/// opaque handles with generation counters for stale reference detection.
class Vmm {
public:
    /// Construct VMM from low-level configuration.
    /// Opens .nxp file, initializes slab allocator, CLOCK-Pro, page table.
    explicit Vmm(VmmConfig config);
    ~Vmm();

    Vmm(const Vmm&) = delete;
    Vmm& operator=(const Vmm&) = delete;

    // Move support for unique_ptr usage
    Vmm(Vmm&& other) noexcept;
    Vmm& operator=(Vmm&& other) noexcept;

    /// Factory: opens .nxp, reads header for ModelParams, computes budget,
    /// validates, and creates Vmm. Returns nullptr if budget is insufficient
    /// (prints report to stderr).
    static std::unique_ptr<Vmm> create(const VmmFullConfig& config);

    /// Get handle for expert by layer/expert ID.
    /// Returns INVALID_HANDLE if out of bounds.
    ExpertHandle get_handle(uint32_t layer_id, uint32_t expert_id) const;

    /// Pin expert: blocks until RESIDENT, returns pointer to weight data.
    /// Returns nullptr if handle is stale (generation mismatch) or load fails.
    const uint8_t* pin(ExpertHandle handle);

    /// Unpin expert: decrements refcount. Transitions CACHED->RESIDENT when
    /// refcount reaches 0.
    void unpin(ExpertHandle handle);

    /// Get current page state for a handle.
    PageState page_state(ExpertHandle handle) const;

    /// Number of pages in RESIDENT or CACHED state.
    size_t resident_count() const;

    /// Number of pages in CACHED state (pinned).
    size_t cached_count() const;

    /// Number of times an expert was loaded from disk (for testing).
    size_t load_count() const;

    /// Runtime statistics snapshot.
    VmmStats stats() const;

    /// Access the computed budget partition (for downstream consumers).
    const BudgetPartition& budget() const;

    /// KV cache base address (pre-allocated, contiguous, 64B-aligned).
    void* kv_cache_base() const;

    /// KV cache size in bytes.
    size_t kv_cache_size() const;

private:
    VmmImpl* impl_;  ///< PIMPL to hide internal details
};

}  // namespace nos
