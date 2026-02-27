/// @file slab_allocator.cpp
/// @brief Fixed-size slab allocator with 64-byte aligned slots.
///
/// Pre-allocates a contiguous arena via aligned_alloc(64, ...) at construction.
/// Allocation and deallocation are O(1) via a stack-based free list.

#include "slab_allocator.h"

#include <cassert>
#include <cstdlib>

namespace nos {

size_t SlabAllocator::align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

SlabAllocator::SlabAllocator(size_t slot_count, size_t slot_size)
    : arena_(nullptr, &::free)
    , slot_size_(align_up(slot_size, 64))  // Ensure 64B alignment per slot
    , slot_count_(slot_count) {
    size_t total = slot_count_ * slot_size_;

    // aligned_alloc requires size to be a multiple of alignment.
    // slot_size_ is already a multiple of 64, and slot_count_ * slot_size_ is too.
    void* mem = ::aligned_alloc(64, total);
    assert(mem && "aligned_alloc failed for slab arena");
    arena_.reset(static_cast<uint8_t*>(mem));

    // Initialize free list: all slots available, LIFO order (low indices first out)
    free_list_.reserve(slot_count_);
    for (int i = static_cast<int>(slot_count_) - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

std::pair<uint8_t*, int> SlabAllocator::allocate() {
    if (free_list_.empty()) {
        return {nullptr, -1};
    }
    int idx = free_list_.back();
    free_list_.pop_back();
    uint8_t* ptr = arena_.get() + static_cast<size_t>(idx) * slot_size_;
    return {ptr, idx};
}

void SlabAllocator::free(int slot_index) {
    assert(slot_index >= 0 &&
           static_cast<size_t>(slot_index) < slot_count_ &&
           "slot_index out of range");
    free_list_.push_back(slot_index);
}

void* SlabAllocator::arena_base() const {
    return arena_.get();
}

size_t SlabAllocator::arena_size() const {
    return slot_count_ * slot_size_;
}

size_t SlabAllocator::slot_size() const {
    return slot_size_;
}

size_t SlabAllocator::available() const {
    return free_list_.size();
}

size_t SlabAllocator::capacity() const {
    return slot_count_;
}

}  // namespace nos
