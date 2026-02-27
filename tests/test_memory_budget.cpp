/// @file test_memory_budget.cpp
/// @brief Unit tests for the memory budget formula: compute_budget(),
///        parse_memory_string(), format_budget_report().

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "vmm/memory_budget.h"

#include <cmath>
#include <cstddef>
#include <string>

using Catch::Matchers::ContainsSubstring;

namespace {

/// Helper: construct typical 70B model params.
nos::ModelParams make_70b_params() {
    nos::ModelParams p{};
    p.n_layers          = 80;
    p.n_kv_heads        = 8;
    p.head_dim          = 128;
    p.hidden_dim        = 8192;
    p.max_expert_size   = 64 * 1024 * 1024;   // 64 MB
    p.top_k             = 4;
    p.experts_per_layer = 8;
    return p;
}

/// Helper: construct typical 7B model params.
nos::ModelParams make_7b_params() {
    nos::ModelParams p{};
    p.n_layers          = 32;
    p.n_kv_heads        = 8;
    p.head_dim          = 128;
    p.hidden_dim        = 4096;
    p.max_expert_size   = 32 * 1024 * 1024;   // 32 MB
    p.top_k             = 2;
    p.experts_per_layer = 8;
    return p;
}

constexpr size_t GB = size_t(1) << 30;
constexpr size_t MB = size_t(1) << 20;

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Correct partition for 16GB budget, 70B model
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: correct partition for 16GB, 70B model", "[memory_budget]") {
    auto p = make_70b_params();

    // kv_per_token = 2 * 80 * 8 * 128 * 2 = 327,680 bytes
    size_t kv_per_token = 2ULL * p.n_layers * p.n_kv_heads * p.head_dim * sizeof(uint16_t);
    REQUIRE(kv_per_token == 327680);

    auto bp = nos::compute_budget(16 * GB, p, 4096);

    CHECK(bp.total == 16 * GB);
    CHECK(bp.sufficient == true);

    // OS overhead = 512 MB
    CHECK(bp.os_overhead == 512 * MB);

    // KV cache = 327680 * 4096 = 1,342,177,280 bytes (~1.25 GB)
    CHECK(bp.kv_cache == kv_per_token * 4096);

    // Working buffers = align_up(8192 * 4 * 3 * 2, 64)
    size_t expected_working = (static_cast<size_t>(p.hidden_dim) * sizeof(float) * 3 * 2 + 63) & ~size_t(63);
    CHECK(bp.working_buffers == expected_working);

    // expert_cache = total - os_overhead - working_buffers - kv_cache
    CHECK(bp.expert_cache == bp.total - bp.os_overhead - bp.working_buffers - bp.kv_cache);

    // expert_slots = expert_cache / max_expert_size
    CHECK(bp.expert_slots == static_cast<uint32_t>(bp.expert_cache / p.max_expert_size));

    // Sum should reconstruct total (within alignment)
    size_t sum = bp.expert_cache + bp.kv_cache + bp.working_buffers + bp.os_overhead;
    CHECK(sum <= bp.total);
    // Expert cache doesn't consume the fractional slot remainder
    CHECK(bp.total - sum < p.max_expert_size);

    INFO("expert_slots = " << bp.expert_slots);
    CHECK(bp.expert_slots > 8);  // 70B model top_k=4, min=8; should be way more with 16GB
}

// ═══════════════════════════════════════════════════════════════════════════
// Correct partition for 8GB budget, 7B model
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: correct partition for 8GB, 7B model", "[memory_budget]") {
    auto p = make_7b_params();
    auto bp = nos::compute_budget(8 * GB, p, 2048);

    CHECK(bp.total == 8 * GB);
    CHECK(bp.sufficient == true);

    // kv_per_token = 2 * 32 * 8 * 128 * 2 = 131,072
    size_t kv_per_token = 2ULL * p.n_layers * p.n_kv_heads * p.head_dim * sizeof(uint16_t);
    CHECK(bp.kv_cache == kv_per_token * 2048);

    CHECK(bp.os_overhead == 512 * MB);
    CHECK(bp.expert_cache > 0);
    CHECK(bp.expert_slots > 0);
    CHECK(bp.expert_slots == static_cast<uint32_t>(bp.expert_cache / p.max_expert_size));
}

// ═══════════════════════════════════════════════════════════════════════════
// Insufficient budget
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: insufficient budget for 70B model", "[memory_budget]") {
    auto p = make_70b_params();
    auto bp = nos::compute_budget(1 * GB, p, 4096);

    CHECK(bp.sufficient == false);
    CHECK(bp.minimum_required > 1 * GB);
    CHECK(bp.minimum_required > 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimum expert slots boundary
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: minimum expert slots boundary", "[memory_budget]") {
    // Use a small model so we can precisely control the boundary.
    nos::ModelParams p{};
    p.n_layers          = 4;
    p.n_kv_heads        = 4;
    p.head_dim          = 64;
    p.hidden_dim        = 256;
    p.max_expert_size   = 1 * MB;
    p.top_k             = 2;
    p.experts_per_layer = 4;

    // kv_per_token = 2 * 4 * 4 * 64 * 2 = 4096
    size_t kv_per_token = 2ULL * p.n_layers * p.n_kv_heads * p.head_dim * sizeof(uint16_t);
    size_t os_overhead = 512 * MB;
    size_t working = (static_cast<size_t>(p.hidden_dim) * sizeof(float) * 3 * 2 + 63) & ~size_t(63);
    size_t kv_cache = kv_per_token * 512;  // minimal context

    uint32_t min_expert_slots = p.top_k * 2;  // = 4

    // Budget that gives exactly min_expert_slots - 1 = 3 slots of expert cache
    size_t budget_insufficient = os_overhead + working + kv_cache + (min_expert_slots - 1) * p.max_expert_size;
    auto bp1 = nos::compute_budget(budget_insufficient, p, 512);
    CHECK(bp1.sufficient == false);

    // Budget that gives exactly min_expert_slots = 4 slots
    size_t budget_sufficient = os_overhead + working + kv_cache + min_expert_slots * p.max_expert_size;
    auto bp2 = nos::compute_budget(budget_sufficient, p, 512);
    CHECK(bp2.sufficient == true);
    CHECK(bp2.expert_slots >= min_expert_slots);
}

// ═══════════════════════════════════════════════════════════════════════════
// Max context computation
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: max context computation", "[memory_budget]") {
    nos::ModelParams p{};
    p.n_layers          = 4;
    p.n_kv_heads        = 4;
    p.head_dim          = 64;
    p.hidden_dim        = 256;
    p.max_expert_size   = 1 * MB;
    p.top_k             = 2;
    p.experts_per_layer = 4;

    size_t kv_per_token = 2ULL * p.n_layers * p.n_kv_heads * p.head_dim * sizeof(uint16_t);
    size_t os_overhead = 512 * MB;
    size_t working = (static_cast<size_t>(p.hidden_dim) * sizeof(float) * 3 * 2 + 63) & ~size_t(63);
    uint32_t min_expert_slots = p.top_k * 2;
    size_t min_expert_cache = static_cast<size_t>(min_expert_slots) * p.max_expert_size;

    // Budget = overhead + working + min_expert + kv for 1000 tokens
    size_t budget = os_overhead + working + min_expert_cache + kv_per_token * 1000;
    auto bp = nos::compute_budget(budget, p, 512);

    // max_context should be floor((budget - os - working - min_expert) / kv_per_token)
    uint32_t expected_max = static_cast<uint32_t>(
        (budget - os_overhead - working - min_expert_cache) / kv_per_token);
    CHECK(bp.max_context == expected_max);
}

// ═══════════════════════════════════════════════════════════════════════════
// Context warning threshold
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: context warning when max_context < 2048", "[memory_budget]") {
    nos::ModelParams p{};
    p.n_layers          = 4;
    p.n_kv_heads        = 4;
    p.head_dim          = 64;
    p.hidden_dim        = 256;
    p.max_expert_size   = 1 * MB;
    p.top_k             = 2;
    p.experts_per_layer = 4;

    size_t kv_per_token = 2ULL * p.n_layers * p.n_kv_heads * p.head_dim * sizeof(uint16_t);
    size_t os_overhead = 512 * MB;
    size_t working = (static_cast<size_t>(p.hidden_dim) * sizeof(float) * 3 * 2 + 63) & ~size_t(63);
    uint32_t min_expert_slots = p.top_k * 2;
    size_t min_expert_cache = static_cast<size_t>(min_expert_slots) * p.max_expert_size;

    // Give a budget that yields max_context ~ 1024 tokens
    size_t budget = os_overhead + working + min_expert_cache + kv_per_token * 1024;
    auto bp = nos::compute_budget(budget, p, 512);
    REQUIRE(bp.sufficient == true);

    std::string report = nos::format_budget_report(bp, p);
    CHECK_THAT(report, ContainsSubstring("WARNING"));
}

// ═══════════════════════════════════════════════════════════════════════════
// parse_memory_string
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: parse_memory_string", "[memory_budget]") {
    CHECK(nos::parse_memory_string("8G") == 8ULL * GB);
    CHECK(nos::parse_memory_string("8GB") == 8ULL * GB);
    CHECK(nos::parse_memory_string("512M") == 512ULL * MB);
    CHECK(nos::parse_memory_string("512MB") == 512ULL * MB);
    CHECK(nos::parse_memory_string("8g") == 8ULL * GB);     // case-insensitive
    CHECK(nos::parse_memory_string("8gb") == 8ULL * GB);    // case-insensitive
    CHECK(nos::parse_memory_string("512m") == 512ULL * MB); // case-insensitive
    CHECK(nos::parse_memory_string("512mb") == 512ULL * MB);
    CHECK(nos::parse_memory_string("abc") == 0);
    CHECK(nos::parse_memory_string("") == 0);
    CHECK(nos::parse_memory_string("16G") == 16ULL * GB);
}

// ═══════════════════════════════════════════════════════════════════════════
// format_budget_report
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Budget: format_budget_report for sufficient budget", "[memory_budget]") {
    auto p = make_70b_params();
    auto bp = nos::compute_budget(16 * GB, p, 4096);
    REQUIRE(bp.sufficient == true);

    std::string report = nos::format_budget_report(bp, p);
    CHECK_THAT(report, ContainsSubstring("Expert cache:"));
    CHECK_THAT(report, ContainsSubstring("KV cache:"));
    CHECK_THAT(report, ContainsSubstring("Max context:"));
    CHECK_THAT(report, ContainsSubstring("OK"));
}

TEST_CASE("Budget: format_budget_report for insufficient budget", "[memory_budget]") {
    auto p = make_70b_params();
    auto bp = nos::compute_budget(1 * GB, p, 4096);
    REQUIRE(bp.sufficient == false);

    std::string report = nos::format_budget_report(bp, p);
    CHECK_THAT(report, ContainsSubstring("INSUFFICIENT"));
    CHECK_THAT(report, ContainsSubstring("Minimum needed:"));
}
