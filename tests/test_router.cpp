#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "engine/router.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("Router: top-2 selects correct experts", "[router]") {
    // 4 experts, hidden_dim=4
    // Expert 0: weights = [1, 0, 0, 0]
    // Expert 1: weights = [0, 1, 0, 0]
    // Expert 2: weights = [0, 0, 1, 0]
    // Expert 3: weights = [0, 0, 0, 1]
    float weights[] = {
        1.0f, 0.0f, 0.0f, 0.0f,   // expert 0
        0.0f, 1.0f, 0.0f, 0.0f,   // expert 1
        0.0f, 0.0f, 1.0f, 0.0f,   // expert 2
        0.0f, 0.0f, 0.0f, 1.0f,   // expert 3
    };

    nos::Router router;
    router.load(weights, 4, 4);

    // Hidden state strongly aligns with expert 2 and somewhat with expert 0
    float hidden[] = {0.3f, 0.0f, 1.0f, 0.1f};

    auto result = router.route(hidden, 2);

    REQUIRE(result.expert_ids.size() == 2);
    REQUIRE(result.gates.size() == 2);

    // Top expert should be expert 2 (score=1.0), then expert 0 (score=0.3)
    REQUIRE(result.expert_ids[0] == 2);
    REQUIRE(result.expert_ids[1] == 0);
}

TEST_CASE("Router: gates sum to 1.0", "[router]") {
    float weights[] = {
        1.0f, 0.5f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.5f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.5f,
        0.5f, 0.0f, 0.0f, 1.0f,
    };

    nos::Router router;
    router.load(weights, 4, 4);

    float hidden[] = {0.5f, 0.5f, 0.5f, 0.5f};
    auto result = router.route(hidden, 2);

    float gate_sum = 0.0f;
    for (float g : result.gates) gate_sum += g;
    REQUIRE_THAT(gate_sum, WithinAbs(1.0f, 1e-5));
}

TEST_CASE("Router: softmax renormalization with different scores", "[router]") {
    float weights[] = {
        10.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 5.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.1f,
    };

    nos::Router router;
    router.load(weights, 4, 4);

    float hidden[] = {1.0f, 1.0f, 1.0f, 1.0f};
    auto result = router.route(hidden, 2);

    // Top-2: expert 0 (score=10) and expert 1 (score=5)
    REQUIRE(result.expert_ids[0] == 0);
    REQUIRE(result.expert_ids[1] == 1);

    // Expert 0 should have higher gate weight than expert 1
    REQUIRE(result.gates[0] > result.gates[1]);

    // Gates still sum to 1
    REQUIRE_THAT(result.gates[0] + result.gates[1], WithinAbs(1.0f, 1e-5));
}

TEST_CASE("Router: top-1 returns single expert", "[router]") {
    float weights[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    nos::Router router;
    router.load(weights, 2, 2);

    float hidden[] = {0.0f, 5.0f};
    auto result = router.route(hidden, 1);

    REQUIRE(result.expert_ids.size() == 1);
    REQUIRE(result.expert_ids[0] == 1);
    REQUIRE_THAT(result.gates[0], WithinAbs(1.0f, 1e-5));
}

TEST_CASE("Router: all gates are positive", "[router]") {
    float weights[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.4f, 0.3f, 0.2f, 0.1f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };

    nos::Router router;
    router.load(weights, 3, 4);

    float hidden[] = {1.0f, 1.0f, 1.0f, 1.0f};
    auto result = router.route(hidden, 3);

    for (float g : result.gates) {
        REQUIRE(g > 0.0f);
    }
}
