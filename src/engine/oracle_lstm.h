#pragma once

/// @file oracle_lstm.h
/// @brief Small LSTM oracle for expert routing prediction.
///
/// Implements a shared-weight LSTM with per-layer hidden/cell state, Glorot
/// uniform initialization, online SGD on W_out only, and nlohmann/json weight
/// persistence. No VMM or engine dependency -- standalone header/implementation.

#include <cstdint>
#include <string>
#include <vector>

namespace nos {

/// Weights shared across all transformer layers.
/// One LstmWeights instance serves all layers; per-layer state is in LstmLayerState.
struct LstmWeights {
    int input_dim  = 0;    ///< num_experts + proj_dim (proj_dim=32)
    int hidden_dim = 0;    ///< 64
    int output_dim = 0;    ///< num_experts * max_k
    int max_k      = 0;    ///< lookahead depth (10)
    int num_experts = 0;   ///< e.g. 64

    /// Combined W_ifgo: [4*hidden_dim, input_dim+hidden_dim] row-major
    std::vector<float> W;       ///< 4*H*(input_dim+H) floats
    std::vector<float> b;       ///< 4*H floats

    /// Output projection
    std::vector<float> W_out;   ///< output_dim * hidden_dim floats
    std::vector<float> b_out;   ///< output_dim floats

    /// Hidden state compression (hidden_state_engine -> proj_dim)
    std::vector<float> W_proj;  ///< proj_dim * (input from engine hidden state) floats
    std::vector<float> b_proj;  ///< proj_dim floats

    void save(const std::string& path) const;
    bool load(const std::string& path);
};

/// Per-transformer-layer LSTM hidden and cell state.
struct LstmLayerState {
    std::vector<float> h;   ///< hidden_dim floats
    std::vector<float> c;   ///< hidden_dim floats

    /// Zero out h and c to hidden_dim size.
    void reset(int hidden_dim);
};

/// Small LSTM oracle: predicts top-k expert IDs K=1..max_k layers ahead.
///
/// Single shared weight matrix; each transformer layer carries its own h/c state.
/// Forward pass is allocation-free (pre-allocated scratch buffers).
/// Online SGD updates W_out only, lr=1e-4, gradient clip norm 1.0, timer-gated at 4ms.
class LstmOracle {
public:
    /// Prediction for one forward_step call: expert IDs and confidence scores per depth.
    struct PredictionResult {
        /// expert_ids[d] = ranked expert IDs for depth d+1 (d in [0..max_k))
        std::vector<std::vector<uint32_t>> expert_ids;
        /// confidences[d] = softmax probabilities matching expert_ids[d]
        std::vector<std::vector<float>>    confidences;
    };

    /// @param num_experts  Number of experts per transformer layer.
    /// @param n_layers     Number of transformer layers.
    /// @param hidden_dim   LSTM hidden size (default 64).
    /// @param max_k        Lookahead depth (default 10).
    /// @param proj_dim     Engine hidden-state compression dim (default 32).
    LstmOracle(int num_experts, int n_layers,
               int hidden_dim = 64, int max_k = 10, int proj_dim = 32);

    /// Initialize weights with Glorot uniform. Call once before use.
    void init_weights();

    /// Forward step for one token at transformer layer @p layer.
    ///
    /// @param layer            Transformer layer index [0..n_layers).
    /// @param router_logits    num_experts floats (raw MoE gating scores).
    /// @param hidden_state     Engine hidden state before MoE (any size).
    /// @param hidden_state_dim Size of hidden_state vector.
    /// @returns Predictions for depths 1..max_k at layer.
    PredictionResult forward_step(int layer, const float* router_logits,
                                  const float* hidden_state, int hidden_state_dim);

    /// Online SGD update: adjusts W_out using actual routing at layer/depth.
    ///
    /// Only fires every UPDATE_INTERVAL tokens and is timer-gated at 4ms.
    /// @param layer           Transformer layer index.
    /// @param depth           1-indexed depth (ground truth is for layer+depth).
    /// @param actual_experts  Expert IDs actually chosen (ground truth).
    void online_update(int layer, int depth,
                       const std::vector<uint32_t>& actual_experts);

    /// Zero out LSTM state for a single layer (call on reset_kv_cache).
    void reset_layer_state(int layer);

    /// Zero out LSTM state for all layers.
    void reset_all_states();

    /// Persist weights to JSON file.
    void save_weights(const std::string& path) const;

    /// Load weights from JSON file.
    /// @returns false if file is missing or dimensions mismatch.
    bool load_weights(const std::string& path);

    /// Approximate RAM usage in bytes (weights + per-layer state + scratch).
    size_t ram_bytes() const;

    /// Online update interval (default 32 tokens).
    static constexpr int UPDATE_INTERVAL = 32;

    // Expose weights for testing (save/load round-trip verification).
    const LstmWeights& weights() const { return weights_; }

private:
    int num_experts_;
    int n_layers_;
    int hidden_dim_;
    int max_k_;
    int proj_dim_;

    LstmWeights weights_;
    std::vector<LstmLayerState> layer_states_;  ///< one per transformer layer

    // Pre-allocated scratch buffers (no heap allocation on the hot path)
    std::vector<float> gate_buf_;    ///< 4 * hidden_dim
    std::vector<float> input_buf_;   ///< input_dim (= num_experts + proj_dim)
    std::vector<float> proj_buf_;    ///< proj_dim
    std::vector<float> out_buf_;     ///< output_dim (= num_experts * max_k)

    /// Per-layer last output (needed for online SGD target computation).
    std::vector<std::vector<float>> last_out_buf_;  ///< [n_layers][output_dim]

    int tokens_since_update_ = 0;
};

}  // namespace nos
