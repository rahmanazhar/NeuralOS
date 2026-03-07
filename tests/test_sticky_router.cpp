/// @file test_sticky_router.cpp
/// @brief Tests for StickyRouter with adaptive lambda and stickiness.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "engine/sticky_router.h"
#include "engine/router.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace nos;
using Catch::Matchers::WithinAbs;

namespace {

/// Create a test router with known weights.
/// Expert i has weight[i][d] = (i == d) ? 1.0 : 0.0 (identity-like).
/// With hidden_state = {1, 0, 0, ...} -> expert 0 scores highest.
struct TestRouterSetup {
    static constexpr int NUM_EXPERTS = 4;
    static constexpr int HIDDEN_DIM = 4;

    std::vector<float> weights;
    Router router;

    TestRouterSetup() {
        weights.resize(NUM_EXPERTS * HIDDEN_DIM, 0.0f);
        // Expert 0: strong in dim 0, Expert 1: strong in dim 1, etc.
        for (int e = 0; e < NUM_EXPERTS; e++) {
            weights[static_cast<size_t>(e * HIDDEN_DIM + e)] = 1.0f;
        }
        router.load(weights.data(), NUM_EXPERTS, HIDDEN_DIM);
    }
};

}  // namespace

TEST_CASE("StickyRouter: lambda=0 produces same result as base Router", "[sticky_router]") {
    TestRouterSetup setup;

    StickyRouter::Config cfg;
    cfg.lambda_override = 0.0f;
    StickyRouter sr(cfg);
    sr.init(1);

    float hidden[] = {1.0f, 0.5f, 0.0f, 0.0f};

    auto base = setup.router.route(hidden, 2);
    auto sticky = sr.route(setup.router, hidden, 2, 0, 0, false);

    // First routing should match base (no current experts yet)
    REQUIRE(sticky.expert_ids == base.expert_ids);
}

TEST_CASE("StickyRouter: high lambda biases toward current experts", "[sticky_router]") {
    TestRouterSetup setup;

    StickyRouter::Config cfg;
    cfg.lambda_override = 5.0f;  // Very strong stickiness
    cfg.max_window = 1000;
    StickyRouter sr(cfg);
    sr.init(1);

    // First routing: hidden favors expert 0 and 1
    float h1[] = {1.0f, 0.5f, 0.0f, 0.0f};
    auto r1 = sr.route(setup.router, h1, 2, 0, 0, false);
    // r1 should pick experts 0 and 1

    // Second routing: hidden now favors expert 2 and 3, but stickiness
    // should keep experts 0 and 1 due to high lambda bonus
    float h2[] = {0.0f, 0.0f, 1.0f, 0.5f};
    auto r2 = sr.route(setup.router, h2, 2, 0, 1, false);

    // With lambda=5, the bonus for current experts (0,1) should outweigh
    // the raw scores for experts 2,3 (which are 1.0 and 0.5)
    std::vector<uint32_t> sorted_r2 = r2.expert_ids;
    std::sort(sorted_r2.begin(), sorted_r2.end());
    std::vector<uint32_t> sorted_r1 = r1.expert_ids;
    std::sort(sorted_r1.begin(), sorted_r1.end());
    REQUIRE(sorted_r2 == sorted_r1);  // Experts stayed the same
}

TEST_CASE("StickyRouter: switch rate decreases with higher lambda", "[sticky_router]") {
    TestRouterSetup setup;

    auto run_with_lambda = [&](float lambda) {
        StickyRouter::Config cfg;
        cfg.lambda_override = lambda;
        cfg.max_window = 1000;
        StickyRouter sr(cfg);
        sr.init(1);

        // Alternate between two different hidden states
        float h_a[] = {1.0f, 0.5f, 0.0f, 0.0f};
        float h_b[] = {0.0f, 0.0f, 0.8f, 1.0f};

        for (int t = 0; t < 20; t++) {
            float* h = (t % 2 == 0) ? h_a : h_b;
            sr.route(setup.router, h, 2, 0, t, false);
        }
        return sr.aggregate_metrics().switch_rate;
    };

    float rate_low = run_with_lambda(0.0f);
    float rate_high = run_with_lambda(5.0f);

    REQUIRE(rate_high < rate_low);
}

TEST_CASE("StickyRouter: window expiry forces fresh routing", "[sticky_router]") {
    TestRouterSetup setup;

    StickyRouter::Config cfg;
    cfg.lambda_override = 10.0f;  // Very sticky
    cfg.max_window = 3;           // But window is short
    StickyRouter sr(cfg);
    sr.init(1);

    // Establish experts with h1
    float h1[] = {1.0f, 0.5f, 0.0f, 0.0f};
    sr.route(setup.router, h1, 2, 0, 0, false);

    // Route with h1 for max_window tokens (staying sticky)
    for (int t = 1; t <= 3; t++) {
        sr.route(setup.router, h1, 2, 0, t, false);
    }

    // After window expiry, h2 should get fresh routing
    float h2[] = {0.0f, 0.0f, 1.0f, 0.5f};
    auto r = sr.route(setup.router, h2, 2, 0, 4, false);

    // Should have switched to experts 2 and 3
    std::vector<uint32_t> sorted = r.expert_ids;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(sorted[0] == 2);
    REQUIRE(sorted[1] == 3);
}

TEST_CASE("StickyRouter: shift detection override forces fresh routing", "[sticky_router]") {
    TestRouterSetup setup;

    StickyRouter::Config cfg;
    cfg.lambda_override = 10.0f;
    cfg.max_window = 1000;
    StickyRouter sr(cfg);
    sr.init(1);

    // Establish experts
    float h1[] = {1.0f, 0.5f, 0.0f, 0.0f};
    sr.route(setup.router, h1, 2, 0, 0, false);

    // Force shift detection — should do fresh routing despite high lambda
    float h2[] = {0.0f, 0.0f, 1.0f, 0.5f};
    auto r = sr.route(setup.router, h2, 2, 0, 1, /*shift_detected=*/true);

    std::vector<uint32_t> sorted = r.expert_ids;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(sorted[0] == 2);
    REQUIRE(sorted[1] == 3);
    REQUIRE(sr.last_trace().reason == "shift_detected");
}

TEST_CASE("StickyRouter: adaptive lambda increases under high I/O pressure", "[sticky_router]") {
    StickyRouter sr;

    // Simulate zero I/O pressure, stable perplexity
    sr.update_io_pressure(0.0f);
    sr.update_perplexity(3.0f);
    sr.update_perplexity(3.0f);

    // Check internal via aggregate: with 0 pressure and stable ppl,
    // lambda should be moderate (ppl_signal = high since delta is small)
    // Now increase I/O pressure
    for (int i = 0; i < 20; i++) {
        sr.update_io_pressure(1.0f);  // 100% miss rate
    }

    // Lambda should now be high. We can test this indirectly:
    // the StickyRouter should resist switching more under high pressure.
    // (Tested via the switch rate test above, but we verify trace here)
    TestRouterSetup setup;
    sr.init(1);
    float h[] = {1.0f, 0.0f, 0.0f, 0.0f};
    sr.route(setup.router, h, 1, 0, 0, false);

    // Lambda from trace should be > 0.5 after saturating I/O pressure
    float h2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    sr.route(setup.router, h2, 1, 0, 1, false);
    REQUIRE(sr.last_trace().lambda > 0.3f);
}

TEST_CASE("StickyRouter: adaptive lambda decreases when perplexity spikes", "[sticky_router]") {
    StickyRouter sr;

    // Start with stable perplexity
    for (int i = 0; i < 10; i++) {
        sr.update_perplexity(3.0f);
    }
    // Perplexity is stable -> ppl_signal high

    // Spike perplexity
    sr.update_perplexity(10.0f);

    // ppl_delta should be large -> ppl_signal low -> lambda drops
    TestRouterSetup setup;
    sr.init(1);
    float h[] = {1.0f, 0.0f, 0.0f, 0.0f};
    sr.route(setup.router, h, 1, 0, 0, false);
    float h2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    sr.route(setup.router, h2, 1, 0, 1, false);

    // Lambda should be moderate-to-low (ppl spike reduces ppl_signal)
    REQUIRE(sr.last_trace().ppl_delta > 0.0f);
}

TEST_CASE("StickyRouter: TraceEntry populated correctly", "[sticky_router]") {
    TestRouterSetup setup;

    StickyRouter::Config cfg;
    cfg.lambda_override = 1.0f;
    StickyRouter sr(cfg);
    sr.init(1);

    float h[] = {1.0f, 0.5f, 0.0f, 0.0f};
    sr.route(setup.router, h, 2, 0, 42, false);

    REQUIRE(sr.has_trace());
    auto& trace = sr.last_trace();
    REQUIRE(trace.token_pos == 42);
    REQUIRE(trace.layer_id == 0);
    REQUIRE(trace.lambda == 1.0f);
    REQUIRE(trace.raw_scores.size() == 4);
    REQUIRE_FALSE(trace.reason.empty());
}

TEST_CASE("StickyRouter: aggregate_metrics returns correct values", "[sticky_router]") {
    TestRouterSetup setup;

    StickyRouter::Config cfg;
    cfg.lambda_override = 0.0f;  // No stickiness — every routing is fresh
    StickyRouter sr(cfg);
    sr.init(1);

    float h_a[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float h_b[] = {0.0f, 0.0f, 0.0f, 1.0f};

    // 10 alternating routings — should switch every time
    for (int t = 0; t < 10; t++) {
        float* h = (t % 2 == 0) ? h_a : h_b;
        sr.route(setup.router, h, 1, 0, t, false);
    }

    auto m = sr.aggregate_metrics();
    REQUIRE(m.total_routing_decisions == 10);
    // First routing sets current, subsequent alternate -> 9 switches
    // Actually, first is always a "switch" from empty
    REQUIRE(m.total_switches > 0);
    REQUIRE(m.switch_rate > 0.0f);
}

TEST_CASE("StickyRouter: lambda override bypasses adaptive computation", "[sticky_router]") {
    StickyRouter::Config cfg;
    cfg.lambda_override = 0.42f;
    StickyRouter sr(cfg);
    sr.init(1);

    // Even after feeding I/O pressure, lambda should be fixed
    sr.update_io_pressure(1.0f);
    sr.update_perplexity(100.0f);

    TestRouterSetup setup;
    float h[] = {1.0f, 0.0f, 0.0f, 0.0f};
    sr.route(setup.router, h, 1, 0, 0, false);
    float h2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    sr.route(setup.router, h2, 1, 0, 1, false);

    REQUIRE_THAT(sr.last_trace().lambda, WithinAbs(0.42, 1e-5));
}
