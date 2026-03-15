#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "engine/prefetch_predictor.h"

using namespace nos;

// ── rank_weighted_precision tests ─────────────────────────────────────────────

TEST_CASE("rank_weighted_precision perfect", "[prefetch_predictor]") {
    std::vector<ExpertId> predicted = {0, 1, 2};
    std::vector<ExpertId> actual    = {0, 1, 2};
    float score = rank_weighted_precision(predicted, actual);
    REQUIRE(score == Catch::Approx(1.0f).epsilon(0.001f));
}

TEST_CASE("rank_weighted_precision empty", "[prefetch_predictor]") {
    REQUIRE(rank_weighted_precision({}, {}) == Catch::Approx(0.0f));
    REQUIRE(rank_weighted_precision({0, 1}, {}) == Catch::Approx(0.0f));
    REQUIRE(rank_weighted_precision({}, {0, 1}) == Catch::Approx(0.0f));
}

TEST_CASE("rank_weighted_precision rank matters", "[prefetch_predictor]") {
    // Predicted [9, 0], actual [0]: rank-2 hit => weight 1/2, max = 1/1+1/2
    // score = 0.5 / 1.5
    std::vector<ExpertId> pred1 = {9, 0};
    std::vector<ExpertId> actual = {0};
    float score1 = rank_weighted_precision(pred1, actual);

    // Predicted [0, 9], actual [0]: rank-1 hit => weight 1/1, max = 1/1+1/2
    // score = 1.0 / 1.5
    std::vector<ExpertId> pred2 = {0, 9};
    float score2 = rank_weighted_precision(pred2, actual);

    REQUIRE(score1 < score2);
    // Specifically check the rank-2 case gives ~0.333 (0.5/1.5)
    REQUIRE(score1 == Catch::Approx(0.5f / 1.5f).epsilon(0.001f));
    // And rank-1 case gives ~0.667 (1.0/1.5)
    REQUIRE(score2 == Catch::Approx(1.0f / 1.5f).epsilon(0.001f));
}

// ── NGramPredictor tests ──────────────────────────────────────────────────────

TEST_CASE("NGramPredictor observe and predict", "[prefetch_predictor]") {
    NGramPredictor pred(1, 3);  // 1-gram, 3 layers

    // Observe expert 2 five times, expert 3 twice, at layer 0
    for (int i = 0; i < 5; ++i) {
        pred.observe(0, {2});
    }
    for (int i = 0; i < 2; ++i) {
        pred.observe(0, {3});
    }

    auto result = pred.predict(0, 1);
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] == 2);
}

TEST_CASE("NGramPredictor per-layer isolation", "[prefetch_predictor]") {
    NGramPredictor pred(1, 3);  // 1-gram, 3 layers

    // Observe only at layer 0
    pred.observe(0, {7});
    pred.observe(0, {7});

    // Layer 1 should return empty
    auto result = pred.predict(1, 1);
    REQUIRE(result.empty());
}

// ── MarkovPredictor tests ─────────────────────────────────────────────────────

TEST_CASE("MarkovPredictor transition", "[prefetch_predictor]") {
    MarkovPredictor pred(2);  // 2 layers

    // Observe {1} then observe {2} at layer 0: transition 1->2
    pred.observe(0, {1});
    pred.observe(0, {2});

    auto result = pred.predict(0, 1);
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] == 2);
}

// ── RepeatLastPredictor tests ─────────────────────────────────────────────────

TEST_CASE("RepeatLastPredictor repeat", "[prefetch_predictor]") {
    RepeatLastPredictor pred(3);  // 3 layers

    pred.observe(1, {3, 4});
    auto result = pred.predict(1, 2);
    REQUIRE(result.size() == 2);
    REQUIRE(result[0] == 3);
    REQUIRE(result[1] == 4);
}

// ── MispredictionTracker tests ────────────────────────────────────────────────

TEST_CASE("MispredictionTracker warmup skips gate", "[prefetch_predictor]") {
    // max_k=5, n_layers=2
    MispredictionTracker tracker(5, 2);

    // Record 30 hits and 20 misses at layer 0, depth 1 (< WARMUP_TOKENS=50)
    for (int i = 0; i < 30; ++i) {
        tracker.record_hit(0, 1);
    }
    for (int i = 0; i < 20; ++i) {
        tracker.record_miss(0, 1);
    }

    // update_k: warmup not complete (-1.0 hit rate), so gate passes
    // effective_k should remain at max_k (5)
    tracker.update_k(0);
    REQUIRE(tracker.effective_k(0) == 5);
}

TEST_CASE("MispredictionTracker reduces K on failure", "[prefetch_predictor]") {
    // max_k=3, n_layers=1
    MispredictionTracker tracker(3, 1);

    // Simulate 200 misses at depth 3 (index 2) to bring hit rate below 0.70
    for (int i = 0; i < 200; ++i) {
        tracker.record_miss(0, 3);
    }

    tracker.update_k(0);
    REQUIRE(tracker.effective_k(0) <= 2);
}

TEST_CASE("MispredictionTracker cooldown", "[prefetch_predictor]") {
    MispredictionTracker tracker(5, 2);

    // Force cooldown with 5 tokens
    tracker.start_cooldown(5);
    REQUIRE(tracker.in_cooldown());

    // Tick 5 times, cooldown should end
    tracker.tick_cooldown();
    tracker.tick_cooldown();
    tracker.tick_cooldown();
    tracker.tick_cooldown();
    REQUIRE(tracker.in_cooldown());  // still in cooldown after 4 ticks
    tracker.tick_cooldown();
    REQUIRE_FALSE(tracker.in_cooldown());  // done after 5th tick
}

// ── NGramTable capacity tests ─────────────────────────────────────────────────

TEST_CASE("NGramTable MAX_ENTRIES cap", "[prefetch_predictor]") {
    NGramTable table;

    // Insert 50001 distinct contexts
    for (uint32_t i = 0; i <= 50000; ++i) {
        table.observe(static_cast<ContextKey>(i), static_cast<ExpertId>(i % 256));
    }

    REQUIRE(table.counts.size() <= static_cast<std::size_t>(NGramTable::MAX_ENTRIES));
}
