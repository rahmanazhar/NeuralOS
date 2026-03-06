/// @file test_shift_detector.cpp
/// @brief Tests for ShiftDetector and cosine_similarity.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "engine/shift_detector.h"

#include <cmath>

using namespace nos;
using Catch::Matchers::WithinAbs;

TEST_CASE("ShiftDetector: no shift on stable signals", "[shift_detector]") {
    ShiftDetector sd;

    // First token: always returns detected=false
    auto r0 = sd.evaluate(1.0f, 0.99f, 2.0f);
    REQUIRE_FALSE(r0.detected);

    // Stable signals: same entropy, high cosine sim, same variance
    auto r1 = sd.evaluate(1.0f, 0.99f, 2.0f);
    REQUIRE_FALSE(r1.detected);
    REQUIRE(r1.confidence < sd.config().threshold);
}

TEST_CASE("ShiftDetector: shift on entropy spike", "[shift_detector]") {
    // Lower threshold so a single strong entropy signal triggers detection
    ShiftDetector::Config cfg;
    cfg.threshold = 0.3f;
    ShiftDetector sd(cfg);

    sd.evaluate(0.5f, 0.99f, 2.0f);  // first token

    // Large entropy increase
    auto r = sd.evaluate(3.0f, 0.99f, 2.0f);
    REQUIRE(r.detected);
    REQUIRE(r.entropy_delta > 0.0f);
}

TEST_CASE("ShiftDetector: shift on cosine drop", "[shift_detector]") {
    ShiftDetector::Config cfg;
    cfg.threshold = 0.2f;
    ShiftDetector sd(cfg);

    sd.evaluate(1.0f, 0.99f, 2.0f);  // first token

    // Very low cosine similarity
    auto r = sd.evaluate(1.0f, 0.05f, 2.0f);
    REQUIRE(r.detected);
}

TEST_CASE("ShiftDetector: shift on variance change", "[shift_detector]") {
    ShiftDetector::Config cfg;
    cfg.threshold = 0.2f;
    ShiftDetector sd(cfg);

    sd.evaluate(1.0f, 0.99f, 2.0f);  // first token

    // Large variance change
    auto r = sd.evaluate(1.0f, 0.99f, 10.0f);
    REQUIRE(r.detected);
}

TEST_CASE("ShiftDetector: multi-signal fusion exceeds cautious threshold", "[shift_detector]") {
    // Default cautious threshold (0.6) requires multiple signals
    ShiftDetector sd;

    sd.evaluate(0.5f, 0.99f, 2.0f);  // first token

    // All signals fire: entropy spike + low cosine + variance change
    auto r = sd.evaluate(3.0f, 0.1f, 10.0f);
    REQUIRE(r.detected);
    REQUIRE(r.confidence > sd.config().threshold);
}

TEST_CASE("ShiftDetector: cooldown suppresses detection", "[shift_detector]") {
    ShiftDetector::Config cfg;
    cfg.cooldown_min = 3;
    cfg.cooldown_max = 10;
    ShiftDetector sd(cfg);

    sd.evaluate(0.5f, 0.99f, 2.0f);  // first token

    // Trigger a shift
    auto r1 = sd.evaluate(3.0f, 0.05f, 10.0f);
    REQUIRE(r1.detected);

    // During cooldown: even strong signals should not detect
    for (int i = 0; i < cfg.cooldown_min; i++) {
        auto r = sd.evaluate(5.0f, 0.01f, 15.0f);
        REQUIRE_FALSE(r.detected);
    }
}

TEST_CASE("ShiftDetector: adaptive cooldown — high confidence = short", "[shift_detector]") {
    ShiftDetector::Config cfg;
    cfg.cooldown_min = 2;
    cfg.cooldown_max = 16;
    ShiftDetector sd(cfg);

    int short_cd = sd.cooldown_tokens(1.0f);  // high confidence
    int long_cd = sd.cooldown_tokens(0.0f);   // low confidence

    REQUIRE(short_cd == cfg.cooldown_min);
    REQUIRE(long_cd == cfg.cooldown_max);

    int mid_cd = sd.cooldown_tokens(0.5f);
    REQUIRE(mid_cd > short_cd);
    REQUIRE(mid_cd < long_cd);
}

TEST_CASE("ShiftDetector: ablation — entropy_weight=0 ignores entropy", "[shift_detector]") {
    ShiftDetector::Config cfg;
    cfg.entropy_weight = 0.0f;
    cfg.cosine_weight = 0.0f;
    cfg.variance_weight = 1.0f;
    cfg.threshold = 0.5f;
    ShiftDetector sd(cfg);

    sd.evaluate(0.5f, 0.99f, 2.0f);  // first token

    // Large entropy change but entropy weight is 0 — only variance matters
    // Same variance, so no shift
    auto r = sd.evaluate(10.0f, 0.99f, 2.0f);
    REQUIRE_FALSE(r.detected);
}

TEST_CASE("ShiftDetector: cold start returns detected=false", "[shift_detector]") {
    ShiftDetector sd;

    // Even with extreme signals, first token returns false
    auto r = sd.evaluate(100.0f, 0.0f, 100.0f);
    REQUIRE_FALSE(r.detected);
    REQUIRE(r.confidence == 0.0f);
}

TEST_CASE("ShiftDetector: reset restores cold start behavior", "[shift_detector]") {
    ShiftDetector sd;

    sd.evaluate(1.0f, 0.99f, 2.0f);
    auto r1 = sd.evaluate(5.0f, 0.05f, 10.0f);
    REQUIRE(r1.detected);

    sd.reset();

    // After reset, first evaluation returns false again
    auto r2 = sd.evaluate(100.0f, 0.0f, 100.0f);
    REQUIRE_FALSE(r2.detected);
}

TEST_CASE("cosine_similarity: identical vectors = 1.0", "[shift_detector]") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    REQUIRE_THAT(cosine_similarity(a, b, 3), WithinAbs(1.0, 1e-5));
}

TEST_CASE("cosine_similarity: orthogonal vectors = 0.0", "[shift_detector]") {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    REQUIRE_THAT(cosine_similarity(a, b, 2), WithinAbs(0.0, 1e-5));
}

TEST_CASE("cosine_similarity: opposite vectors = -1.0", "[shift_detector]") {
    float a[] = {1.0f, 2.0f};
    float b[] = {-1.0f, -2.0f};
    REQUIRE_THAT(cosine_similarity(a, b, 2), WithinAbs(-1.0, 1e-5));
}

TEST_CASE("cosine_similarity: zero vector returns 0.0", "[shift_detector]") {
    float a[] = {0.0f, 0.0f};
    float b[] = {1.0f, 2.0f};
    REQUIRE_THAT(cosine_similarity(a, b, 2), WithinAbs(0.0, 1e-5));
}
