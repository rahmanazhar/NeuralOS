#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "engine/sampling.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinAbs;

TEST_CASE("Sampling: greedy returns argmax", "[sampling]") {
    float logits[] = {1.0f, 5.0f, 2.0f, 0.5f, 3.0f};
    REQUIRE(nos::greedy_sample(logits, 5) == 1);
}

TEST_CASE("Sampling: temperature scaling", "[sampling]") {
    float logits[] = {1.0f, 2.0f, 3.0f};
    nos::apply_temperature(logits, 3, 0.5f);
    // 1/0.5 = 2x scaling
    REQUIRE_THAT(logits[0], WithinAbs(2.0f, 1e-5));
    REQUIRE_THAT(logits[1], WithinAbs(4.0f, 1e-5));
    REQUIRE_THAT(logits[2], WithinAbs(6.0f, 1e-5));
}

TEST_CASE("Sampling: top-k filtering", "[sampling]") {
    float logits[] = {1.0f, 5.0f, 2.0f, 4.0f, 3.0f};
    nos::apply_top_k(logits, 5, 2);
    // Top 2 are indices 1 (5.0) and 3 (4.0)
    REQUIRE(logits[1] == 5.0f);
    REQUIRE(logits[3] == 4.0f);
    REQUIRE(logits[0] == -std::numeric_limits<float>::infinity());
    REQUIRE(logits[2] == -std::numeric_limits<float>::infinity());
    REQUIRE(logits[4] == -std::numeric_limits<float>::infinity());
}

TEST_CASE("Sampling: top-p nucleus filtering", "[sampling]") {
    // logits [10, 5, 0, 0, 0] -> after softmax, first token has ~99% prob
    float logits[] = {10.0f, 5.0f, 0.0f, 0.0f, 0.0f};
    nos::apply_top_p(logits, 5, 0.5f);
    // Token 0 should survive (its prob alone exceeds p=0.5)
    REQUIRE(logits[0] == 10.0f);
    // Some tokens should be filtered out
    int filtered = 0;
    for (int i = 0; i < 5; i++) {
        if (logits[i] == -std::numeric_limits<float>::infinity()) filtered++;
    }
    REQUIRE(filtered > 0);
}

TEST_CASE("Sampling: min-p filtering", "[sampling]") {
    float logits[] = {10.0f, 5.0f, 0.0f, -5.0f, -10.0f};
    nos::apply_min_p(logits, 5, 0.1f);
    // threshold = 10.0 + log(0.1) = 10.0 - 2.302... = 7.697...
    // Only logits >= ~7.7 survive
    REQUIRE(logits[0] == 10.0f);
    REQUIRE(logits[1] == -std::numeric_limits<float>::infinity());
}

TEST_CASE("Sampling: repetition penalty", "[sampling]") {
    float logits[] = {2.0f, -1.0f, 3.0f, 0.5f};
    std::vector<int> context = {0, 1};  // penalize tokens 0 and 1
    nos::apply_repetition_penalty(logits, 4, context, 2.0f);
    // Token 0 (positive): 2.0 / 2.0 = 1.0
    REQUIRE_THAT(logits[0], WithinAbs(1.0f, 1e-5));
    // Token 1 (negative): -1.0 * 2.0 = -2.0
    REQUIRE_THAT(logits[1], WithinAbs(-2.0f, 1e-5));
    // Tokens not in context: unchanged
    REQUIRE_THAT(logits[2], WithinAbs(3.0f, 1e-5));
    REQUIRE_THAT(logits[3], WithinAbs(0.5f, 1e-5));
}

TEST_CASE("Sampling: full pipeline produces valid token", "[sampling]") {
    float logits[] = {1.0f, 5.0f, 2.0f, 0.5f, 3.0f};
    nos::SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 3;
    params.top_p = 0.95f;
    params.repetition_penalty = 1.0f;  // disabled
    params.min_p = 0.0f;              // disabled
    params.seed = 42;
    std::vector<int> context = {0};
    std::mt19937 rng(42);

    int token = nos::sample(logits, 5, params, context, rng);
    REQUIRE(token >= 0);
    REQUIRE(token < 5);
}

TEST_CASE("Sampling: deterministic with same seed", "[sampling]") {
    auto run_sample = [](uint64_t seed) {
        float logits[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        nos::SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 0;
        params.top_p = 1.0f;
        params.repetition_penalty = 1.0f;
        params.min_p = 0.0f;
        std::vector<int> context;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
        return nos::sample(logits, 8, params, context, rng);
    };

    int result1 = run_sample(123);
    int result2 = run_sample(123);
    REQUIRE(result1 == result2);
}
