#pragma once

/// @file slab_allocator.h
/// @brief Fixed-size slab allocator with 64-byte aligned slots.
///
/// Pre-allocates a contiguous arena of fixed-size slots at startup.
/// Allocation pops from a free list (O(1)), deallocation pushes back (O(1)).
/// All slots are 64-byte aligned for NXP_ALIGNMENT and cache-line compatibility.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace nos {

class SlabAllocator {
public:
    /// Initialize with slot_count slots of slot_size bytes each, 64B aligned.
    SlabAllocator(size_t slot_count, size_t slot_size);

    ~SlabAllocator() = default;

    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    /// Allocate a slot. Returns {pointer, slot_index} or {nullptr, -1} if full.
    std::pair<uint8_t*, int> allocate();

    /// Free a slot by index. O(1).
    void free(int slot_index);

    /// Pointer to arena start (for future io_uring buffer registration).
    void* arena_base() const;

    /// Total arena size in bytes.
    size_t arena_size() const;

    /// Aligned slot size in bytes.
    size_t slot_size() const;

    /// Number of currently available (free) slots.
    size_t available() const;

    /// Total slot count.
    size_t capacity() const;

private:
    /// Round up val to the next multiple of alignment.
    static size_t align_up(size_t val, size_t alignment);

    std::unique_ptr<uint8_t, decltype(&::free)> arena_;
    std::vector<int> free_list_;
    size_t slot_size_;
    size_t slot_count_;
};

}  // namespace nos
