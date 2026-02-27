/// @file test_slab_allocator.cpp
/// @brief Unit tests for the slab allocator.

#include <catch2/catch_test_macros.hpp>

#include "vmm/slab_allocator.h"

#include <cstdint>
#include <set>
#include <vector>

TEST_CASE("SlabAllocator: allocate all slots returns unique non-null pointers",
          "[slab_allocator]") {
    constexpr size_t SLOT_COUNT = 16;
    constexpr size_t SLOT_SIZE = 1024;
    nos::SlabAllocator slab(SLOT_COUNT, SLOT_SIZE);

    std::set<uint8_t*> ptrs;
    std::set<int> indices;

    for (size_t i = 0; i < SLOT_COUNT; ++i) {
        auto [ptr, idx] = slab.allocate();
        REQUIRE(ptr != nullptr);
        REQUIRE(idx >= 0);
        REQUIRE(idx < static_cast<int>(SLOT_COUNT));
        ptrs.insert(ptr);
        indices.insert(idx);
    }

    // All pointers and indices must be unique
    CHECK(ptrs.size() == SLOT_COUNT);
    CHECK(indices.size() == SLOT_COUNT);
}

TEST_CASE("SlabAllocator: 64-byte alignment of every allocated pointer",
          "[slab_allocator]") {
    constexpr size_t SLOT_COUNT = 32;
    constexpr size_t SLOT_SIZE = 256;
    nos::SlabAllocator slab(SLOT_COUNT, SLOT_SIZE);

    for (size_t i = 0; i < SLOT_COUNT; ++i) {
        auto [ptr, idx] = slab.allocate();
        REQUIRE(ptr != nullptr);
        auto addr = reinterpret_cast<uintptr_t>(ptr);
        CHECK((addr % 64) == 0);
    }
}

TEST_CASE("SlabAllocator: free all then reallocate -- same capacity",
          "[slab_allocator]") {
    constexpr size_t SLOT_COUNT = 8;
    constexpr size_t SLOT_SIZE = 512;
    nos::SlabAllocator slab(SLOT_COUNT, SLOT_SIZE);

    // Allocate all
    std::vector<int> slot_indices;
    for (size_t i = 0; i < SLOT_COUNT; ++i) {
        auto [ptr, idx] = slab.allocate();
        REQUIRE(ptr != nullptr);
        slot_indices.push_back(idx);
    }
    CHECK(slab.available() == 0);

    // Free all
    for (int idx : slot_indices) {
        slab.free(idx);
    }
    CHECK(slab.available() == SLOT_COUNT);

    // Reallocate all -- must succeed
    for (size_t i = 0; i < SLOT_COUNT; ++i) {
        auto [ptr, idx] = slab.allocate();
        CHECK(ptr != nullptr);
        CHECK(idx >= 0);
    }
    CHECK(slab.available() == 0);
}

TEST_CASE("SlabAllocator: allocate beyond capacity returns {nullptr, -1}",
          "[slab_allocator]") {
    constexpr size_t SLOT_COUNT = 4;
    constexpr size_t SLOT_SIZE = 128;
    nos::SlabAllocator slab(SLOT_COUNT, SLOT_SIZE);

    // Exhaust
    for (size_t i = 0; i < SLOT_COUNT; ++i) {
        auto [ptr, idx] = slab.allocate();
        REQUIRE(ptr != nullptr);
    }

    // Next allocation must fail
    auto [ptr, idx] = slab.allocate();
    CHECK(ptr == nullptr);
    CHECK(idx == -1);
}

TEST_CASE("SlabAllocator: free and re-allocate single slot (O(1) behavior)",
          "[slab_allocator]") {
    constexpr size_t SLOT_COUNT = 4;
    constexpr size_t SLOT_SIZE = 256;
    nos::SlabAllocator slab(SLOT_COUNT, SLOT_SIZE);

    // Allocate all
    std::vector<std::pair<uint8_t*, int>> allocs;
    for (size_t i = 0; i < SLOT_COUNT; ++i) {
        allocs.push_back(slab.allocate());
    }

    // Free slot 2
    slab.free(allocs[2].second);
    CHECK(slab.available() == 1);

    // Re-allocate -- should get back the freed slot
    auto [ptr, idx] = slab.allocate();
    CHECK(ptr != nullptr);
    CHECK(idx == allocs[2].second);
    CHECK(slab.available() == 0);
}

TEST_CASE("SlabAllocator: arena_base, arena_size, slot_size, capacity getters",
          "[slab_allocator]") {
    constexpr size_t SLOT_COUNT = 10;
    constexpr size_t SLOT_SIZE = 100;  // Not 64-byte aligned
    nos::SlabAllocator slab(SLOT_COUNT, SLOT_SIZE);

    CHECK(slab.arena_base() != nullptr);
    CHECK(slab.capacity() == SLOT_COUNT);

    // slot_size should be aligned up to 64 bytes
    CHECK(slab.slot_size() >= SLOT_SIZE);
    CHECK((slab.slot_size() % 64) == 0);

    // arena_size = slot_count * aligned_slot_size
    CHECK(slab.arena_size() == SLOT_COUNT * slab.slot_size());

    // available starts at capacity
    CHECK(slab.available() == SLOT_COUNT);
}

TEST_CASE("SlabAllocator: slot_size rounds up non-aligned sizes to 64B boundary",
          "[slab_allocator]") {
    // 100 bytes -> aligned up to 128
    nos::SlabAllocator slab1(4, 100);
    CHECK(slab1.slot_size() == 128);

    // 64 bytes -> stays 64
    nos::SlabAllocator slab2(4, 64);
    CHECK(slab2.slot_size() == 64);

    // 65 bytes -> aligned up to 128
    nos::SlabAllocator slab3(4, 65);
    CHECK(slab3.slot_size() == 128);

    // 1 byte -> aligned up to 64
    nos::SlabAllocator slab4(4, 1);
    CHECK(slab4.slot_size() == 64);
}
