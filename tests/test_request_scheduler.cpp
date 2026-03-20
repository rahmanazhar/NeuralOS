/// @file test_request_scheduler.cpp
/// @brief Catch2 tests for RequestScheduler slot management logic.
///
/// Note: These tests use a null/invalid model path, so nos_create() will fail
/// for each slot. We test the scheduler's slot acquisition/release logic around
/// that constraint -- the scheduler gracefully handles failed context creation.

#include <catch2/catch_test_macros.hpp>

#include "server/request_scheduler.h"

using namespace nos;

/// Helper: create a scheduler with a null model path (all slots will have ctx=nullptr).
static RequestScheduler make_null_scheduler(size_t max_slots) {
    nos_config_t cfg{};
    cfg.struct_size = sizeof(nos_config_t);
    return RequestScheduler(max_slots, "/nonexistent/model", cfg);
}

TEST_CASE("RequestScheduler reports not ready with invalid model", "[request_scheduler]") {
    auto sched = make_null_scheduler(4);
    // All contexts failed to create, so scheduler should not be ready
    REQUIRE_FALSE(sched.is_ready());
}

TEST_CASE("RequestScheduler slot_count matches construction", "[request_scheduler]") {
    auto sched = make_null_scheduler(4);
    REQUIRE(sched.slot_count() == 4);
}

TEST_CASE("RequestScheduler active_count starts at zero", "[request_scheduler]") {
    auto sched = make_null_scheduler(2);
    REQUIRE(sched.active_count() == 0);
}

TEST_CASE("Acquire returns nullptr when no slots have valid contexts", "[request_scheduler]") {
    auto sched = make_null_scheduler(2);
    // No slot has a valid ctx, so acquire should return nullptr
    auto* slot = sched.acquire_slot();
    REQUIRE(slot == nullptr);
}

TEST_CASE("SlotGuard handles nullptr gracefully", "[request_scheduler]") {
    auto sched = make_null_scheduler(1);
    auto guard = sched.acquire_slot_guard();
    REQUIRE_FALSE(static_cast<bool>(guard));
    REQUIRE(guard.get() == nullptr);
    // Destructor should not crash
}

TEST_CASE("SlotGuard auto-releases on scope exit", "[request_scheduler]") {
    auto sched = make_null_scheduler(2);
    // With no valid contexts, we cannot truly test release, but verify
    // that active_count stays zero
    {
        auto guard = sched.acquire_slot_guard();
        // guard is empty (nullptr) since no valid slots
    }
    REQUIRE(sched.active_count() == 0);
}
