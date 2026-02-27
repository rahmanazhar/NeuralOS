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
#include <string>

namespace nos {

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

/// Configuration for the VMM.
struct VmmConfig {
    size_t   expert_cache_bytes;    ///< Total bytes for expert slab pool
    size_t   max_expert_size;       ///< Size of largest expert (weights + scales)
    uint32_t num_layers;            ///< Number of model layers
    uint32_t experts_per_layer;     ///< Number of experts per layer
    std::string nxp_path;           ///< Path to .nxp file
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
    /// Construct VMM from configuration.
    /// Opens .nxp file, initializes slab allocator, CLOCK-Pro, page table.
    explicit Vmm(VmmConfig config);
    ~Vmm();

    Vmm(const Vmm&) = delete;
    Vmm& operator=(const Vmm&) = delete;

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

private:
    VmmImpl* impl_;  ///< PIMPL to hide internal details
};

}  // namespace nos
