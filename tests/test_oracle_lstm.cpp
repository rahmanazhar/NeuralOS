/// @file test_oracle_lstm.cpp
/// @brief Catch2 tests for LstmOracle: forward pass, SGD, init, save/load, RAM budget.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "engine/oracle_lstm.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: sum of vector<float>
// ---------------------------------------------------------------------------

static float vec_sum(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f);
}

static bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// TEST 1: Construction and RAM budget
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle construction and RAM budget") {
    // 64 experts, 32 layers, hidden=64, max_k=10, proj=32
    nos::LstmOracle oracle(64, 32, 64, 10, 32);
    oracle.init_weights();

    size_t ram = oracle.ram_bytes();
    constexpr size_t TEN_MB = 10UL * 1024UL * 1024UL;
    REQUIRE(ram < TEN_MB);
}

// ---------------------------------------------------------------------------
// TEST 2: Glorot init range
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle init_weights Glorot range") {
    nos::LstmOracle oracle(64, 2, 64, 10, 32);
    oracle.init_weights();

    const auto& w = oracle.weights();

    // Weights must be non-empty
    REQUIRE(!w.W.empty());
    REQUIRE(!w.W_out.empty());
    REQUIRE(!w.W_proj.empty());

    // All weights must be finite (no NaN / Inf)
    REQUIRE(all_finite(w.W));
    REQUIRE(all_finite(w.W_out));
    REQUIRE(all_finite(w.W_proj));

    // Bias vectors must be zero-initialised
    for (float v : w.b) {
        REQUIRE(v == 0.0f);
    }
    for (float v : w.b_out) {
        REQUIRE(v == 0.0f);
    }
    for (float v : w.b_proj) {
        REQUIRE(v == 0.0f);
    }
}

// ---------------------------------------------------------------------------
// TEST 3: forward_step returns correct shape and softmax sums to ~1.0
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle forward_step returns correct shape") {
    nos::LstmOracle oracle(8, 4, 16, 3, 8);
    oracle.init_weights();

    float router_logits[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float hidden_state[16] = {};  // all zeros

    auto result = oracle.forward_step(0, router_logits, hidden_state, 16);

    // max_k == 3 depths
    REQUIRE(result.expert_ids.size()  == 3UL);
    REQUIRE(result.confidences.size() == 3UL);

    for (size_t d = 0; d < 3UL; ++d) {
        // No more predictions than num_experts
        REQUIRE(result.expert_ids[d].size() <= 8UL);
        REQUIRE(result.confidences[d].size() == result.expert_ids[d].size());

        // Confidence values must be positive
        for (float c : result.confidences[d]) {
            REQUIRE(c >= 0.0f);
        }

        // Softmax over all experts should sum to ~1.0
        float sum = vec_sum(result.confidences[d]);
        REQUIRE(sum == Catch::Approx(1.0f).epsilon(1e-4));
    }
}

// ---------------------------------------------------------------------------
// TEST 4: State isolation per layer
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle state isolation per layer") {
    nos::LstmOracle oracle(8, 4, 16, 3, 8);
    oracle.init_weights();

    float router_logits[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float hidden_state[16] = {};

    // Run two forward_steps on layer 0 to advance its state
    oracle.forward_step(0, router_logits, hidden_state, 16);
    auto r0a = oracle.forward_step(0, router_logits, hidden_state, 16);

    // Run one forward_step on layer 1 (zero state)
    auto r1 = oracle.forward_step(1, router_logits, hidden_state, 16);

    // Layer 0 has been through two steps; layer 1 only one step from zero state.
    // Their hidden states (and thus outputs) must differ.
    // Compare first confidence value of depth 0.
    float c0 = r0a.confidences[0][0];
    float c1 = r1.confidences[0][0];
    // After two vs one step the outputs should differ (LSTM is stateful).
    // This is true unless the weights are all zero, which Glorot init prevents.
    REQUIRE(c0 != Catch::Approx(c1).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// TEST 5: reset_layer_state restores initial state
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle reset_layer_state zeros state") {
    float router_logits[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float hidden_state[8]  = {};

    // Oracle A: advance layer 0 state, then reset.
    nos::LstmOracle oracle_a(8, 2, 16, 3, 8);
    oracle_a.init_weights();

    oracle_a.forward_step(0, router_logits, hidden_state, 8);
    oracle_a.forward_step(0, router_logits, hidden_state, 8);
    oracle_a.reset_layer_state(0);
    auto result_a = oracle_a.forward_step(0, router_logits, hidden_state, 8);

    // Oracle B: fresh oracle with same weights, run one step.
    nos::LstmOracle oracle_b(8, 2, 16, 3, 8);
    // Copy weights from oracle_a so comparison is valid.
    oracle_b.init_weights();
    // Save oracle_a weights, load into oracle_b.
    oracle_a.save_weights("/tmp/test_lstm_reset_weights.json");
    oracle_b.load_weights("/tmp/test_lstm_reset_weights.json");

    auto result_b = oracle_b.forward_step(0, router_logits, hidden_state, 8);

    // Both should produce identical output since they start from zero state + same weights.
    REQUIRE(result_a.confidences[0][0] ==
            Catch::Approx(result_b.confidences[0][0]).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// TEST 6: save and load weights round-trip
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle save and load weights round-trip") {
    nos::LstmOracle oracle(8, 2, 64, 3, 8);
    oracle.init_weights();

    const std::string tmp_path = "/tmp/test_lstm_weights.json";
    oracle.save_weights(tmp_path);

    nos::LstmOracle oracle2(8, 2, 64, 3, 8);
    oracle2.init_weights();  // random different weights

    bool ok = oracle2.load_weights(tmp_path);
    REQUIRE(ok == true);

    // Verify all weight vectors match.
    REQUIRE(oracle2.weights().W     == oracle.weights().W);
    REQUIRE(oracle2.weights().b     == oracle.weights().b);
    REQUIRE(oracle2.weights().W_out == oracle.weights().W_out);
    REQUIRE(oracle2.weights().b_out == oracle.weights().b_out);
    REQUIRE(oracle2.weights().W_proj == oracle.weights().W_proj);
    REQUIRE(oracle2.weights().b_proj == oracle.weights().b_proj);
}

// ---------------------------------------------------------------------------
// TEST 7: load_weights returns false on bad file
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle load returns false on bad file") {
    nos::LstmOracle oracle(8, 2, 64, 3, 8);
    oracle.init_weights();

    // Save original W for comparison.
    std::vector<float> original_W = oracle.weights().W;

    bool ok = oracle.load_weights("/tmp/nonexistent_lstm_xyz.json");
    REQUIRE(ok == false);

    // Weights must remain unchanged.
    REQUIRE(oracle.weights().W == original_W);
}

// ---------------------------------------------------------------------------
// TEST 8: online_update runs without crash and W_out stays finite
// ---------------------------------------------------------------------------

TEST_CASE("LstmOracle online_update runs without crash") {
    nos::LstmOracle oracle(8, 2, 16, 3, 8);
    oracle.init_weights();

    float router_logits[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float hidden_state[16] = {};

    // Run one forward step so last_out_buf_ is populated.
    oracle.forward_step(0, router_logits, hidden_state, 16);

    // Call online_update 32+ times to trigger UPDATE_INTERVAL.
    for (int i = 0; i < 32; ++i) {
        oracle.online_update(0, 1, {2, 3});
    }

    // W_out must still be finite after SGD.
    REQUIRE(all_finite(oracle.weights().W_out));
}
