/// @file oracle_lstm.cpp
/// @brief LstmOracle implementation: forward pass, online SGD, Glorot init, save/load.

#include "engine/oracle_lstm.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace nos {

// ---------------------------------------------------------------------------
// LstmLayerState
// ---------------------------------------------------------------------------

void LstmLayerState::reset(int hidden_dim) {
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };
    h.assign(sz(hidden_dim), 0.0f);
    c.assign(sz(hidden_dim), 0.0f);
}

// ---------------------------------------------------------------------------
// LstmWeights: save / load
// ---------------------------------------------------------------------------

void LstmWeights::save(const std::string& path) const {
    nlohmann::json j;
    j["input_dim"]   = input_dim;
    j["hidden_dim"]  = hidden_dim;
    j["output_dim"]  = output_dim;
    j["max_k"]       = max_k;
    j["num_experts"] = num_experts;
    j["W"]      = W;
    j["b"]      = b;
    j["W_out"]  = W_out;
    j["b_out"]  = b_out;
    j["W_proj"] = W_proj;
    j["b_proj"] = b_proj;

    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("LstmWeights::save: cannot open " + path);
    }
    ofs << j.dump(2);
}

bool LstmWeights::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }

    nlohmann::json j;
    try {
        ifs >> j;
    } catch (...) {
        return false;
    }

    try {
        int new_input  = j.at("input_dim").get<int>();
        int new_hidden = j.at("hidden_dim").get<int>();
        int new_output = j.at("output_dim").get<int>();
        int new_maxk   = j.at("max_k").get<int>();
        int new_nexps  = j.at("num_experts").get<int>();

        // Validate expected sizes match what was serialized.
        auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };
        size_t expected_W      = sz(4) * sz(new_hidden) * sz(new_input + new_hidden);
        size_t expected_b      = sz(4) * sz(new_hidden);
        size_t expected_W_out  = sz(new_output) * sz(new_hidden);
        size_t expected_b_out  = sz(new_output);

        auto new_W      = j.at("W").get<std::vector<float>>();
        auto new_b      = j.at("b").get<std::vector<float>>();
        auto new_W_out  = j.at("W_out").get<std::vector<float>>();
        auto new_b_out  = j.at("b_out").get<std::vector<float>>();
        auto new_W_proj = j.at("W_proj").get<std::vector<float>>();
        auto new_b_proj = j.at("b_proj").get<std::vector<float>>();

        if (new_W.size()     != expected_W    ||
            new_b.size()     != expected_b    ||
            new_W_out.size() != expected_W_out ||
            new_b_out.size() != expected_b_out) {
            return false;
        }

        input_dim   = new_input;
        hidden_dim  = new_hidden;
        output_dim  = new_output;
        max_k       = new_maxk;
        num_experts = new_nexps;
        W      = std::move(new_W);
        b      = std::move(new_b);
        W_out  = std::move(new_W_out);
        b_out  = std::move(new_b_out);
        W_proj = std::move(new_W_proj);
        b_proj = std::move(new_b_proj);
        return true;
    } catch (...) {
        return false;
    }
}

// ---------------------------------------------------------------------------
// LstmOracle: constructor
// ---------------------------------------------------------------------------

LstmOracle::LstmOracle(int num_experts, int n_layers,
                       int hidden_dim, int max_k, int proj_dim)
    : num_experts_(num_experts),
      n_layers_(n_layers),
      hidden_dim_(hidden_dim),
      max_k_(max_k),
      proj_dim_(proj_dim)
{
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };

    // Configure weight dimensions.
    weights_.input_dim   = num_experts + proj_dim;
    weights_.hidden_dim  = hidden_dim;
    weights_.output_dim  = num_experts * max_k;
    weights_.max_k       = max_k;
    weights_.num_experts = num_experts;

    // Allocate per-layer states (zeroed).
    layer_states_.resize(sz(n_layers_));
    for (auto& s : layer_states_) {
        s.reset(hidden_dim_);
    }

    // Pre-allocate scratch buffers.
    gate_buf_.resize(sz(4) * sz(hidden_dim_),       0.0f);
    input_buf_.resize(sz(num_experts_) + sz(proj_dim_), 0.0f);
    proj_buf_.resize(sz(proj_dim_),                 0.0f);
    out_buf_.resize(sz(num_experts_) * sz(max_k_),  0.0f);

    // Per-layer last output buffer.
    last_out_buf_.resize(sz(n_layers_),
                         std::vector<float>(sz(num_experts_) * sz(max_k_), 0.0f));
}

// ---------------------------------------------------------------------------
// LstmOracle::init_weights -- Glorot uniform
// ---------------------------------------------------------------------------

void LstmOracle::init_weights() {
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };

    std::mt19937 rng{std::random_device{}()};

    // Helper: fill vector with Glorot uniform values for shape [rows, cols].
    auto glorot_fill = [&](std::vector<float>& w, int rows, int cols) {
        float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
        std::uniform_real_distribution<float> dist(-limit, limit);
        w.resize(sz(rows) * sz(cols));
        for (auto& v : w) {
            v = dist(rng);
        }
    };

    int H     = hidden_dim_;
    int idim  = weights_.input_dim;   // num_experts + proj_dim
    int odim  = weights_.output_dim;  // num_experts * max_k
    // proj_dim_ columns, hidden_state can be any size; stored as proj_dim_ x proj_dim_
    // For W_proj we don't know the engine hidden state size at init time, so
    // we initialise W_proj as [proj_dim, proj_dim] (identity-compatible square).
    // forward_step will project the first min(hidden_state_dim, proj_dim) dimensions.
    int prows = proj_dim_;
    int pcols = proj_dim_;

    // W: [4*H, idim+H]
    glorot_fill(weights_.W, 4 * H, idim + H);

    // b: [4*H] -- zero init
    weights_.b.assign(sz(4) * sz(H), 0.0f);

    // W_out: [odim, H]
    glorot_fill(weights_.W_out, odim, H);

    // b_out: [odim] -- zero init
    weights_.b_out.assign(sz(odim), 0.0f);

    // W_proj: [proj_dim, proj_dim]
    glorot_fill(weights_.W_proj, prows, pcols);

    // b_proj: [proj_dim] -- zero init
    weights_.b_proj.assign(sz(proj_dim_), 0.0f);
}

// ---------------------------------------------------------------------------
// LstmOracle::forward_step
// ---------------------------------------------------------------------------

LstmOracle::PredictionResult LstmOracle::forward_step(
    int layer,
    const float* router_logits,
    const float* hidden_state,
    int hidden_state_dim)
{
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };

    const int H    = hidden_dim_;
    const int idim = weights_.input_dim;   // num_experts + proj_dim

    // --- 1. Project hidden_state -> proj_buf_ via W_proj * hidden_state + b_proj ---
    // W_proj is [proj_dim, proj_dim]. We use min(hidden_state_dim, proj_dim) input dims.
    const int proj_input_cols = std::min(hidden_state_dim, proj_dim_);
    for (int i = 0; i < proj_dim_; ++i) {
        float sum = weights_.b_proj[sz(i)];
        for (int j = 0; j < proj_input_cols; ++j) {
            sum += weights_.W_proj[sz(i) * sz(proj_dim_) + sz(j)] * hidden_state[sz(j)];
        }
        proj_buf_[sz(i)] = sum;
    }

    // --- 2. Build input_buf_ = concat(router_logits, proj_buf_) ---
    for (int j = 0; j < num_experts_; ++j) {
        input_buf_[sz(j)] = router_logits[sz(j)];
    }
    for (int j = 0; j < proj_dim_; ++j) {
        input_buf_[sz(num_experts_) + sz(j)] = proj_buf_[sz(j)];
    }

    // --- 3. Compute all 4 gates fused: gate_buf_[i] = b[i] + W[i,:] * [h; x] ---
    LstmLayerState& state = layer_states_[sz(layer)];
    const int combined = H + idim;  // size of [h; x] concatenation

    for (int i = 0; i < 4 * H; ++i) {
        float sum = weights_.b[sz(i)];
        // dot over [h(t-1); input_buf_]
        for (int j = 0; j < combined; ++j) {
            float xj = (j < H) ? state.h[sz(j)] : input_buf_[sz(j - H)];
            sum += weights_.W[sz(i) * sz(combined) + sz(j)] * xj;
        }
        gate_buf_[sz(i)] = sum;
    }

    // --- 4. Apply activations and update c, h in-place ---
    // i-gate: gate_buf_[0..H),   sigmoid
    // f-gate: gate_buf_[H..2H),  sigmoid
    // g-gate: gate_buf_[2H..3H), tanh
    // o-gate: gate_buf_[3H..4H), sigmoid
    for (int i = 0; i < H; ++i) {
        float ig = 1.0f / (1.0f + std::exp(-gate_buf_[sz(i)]));
        float fg = 1.0f / (1.0f + std::exp(-gate_buf_[sz(H + i)]));
        float gg = std::tanh(gate_buf_[sz(2 * H + i)]);
        float og = 1.0f / (1.0f + std::exp(-gate_buf_[sz(3 * H + i)]));
        state.c[sz(i)] = fg * state.c[sz(i)] + ig * gg;
        state.h[sz(i)] = og * std::tanh(state.c[sz(i)]);
    }

    // --- 5. Output projection: out_buf_[i] = b_out[i] + W_out[i,:] * h ---
    const int odim = weights_.output_dim;
    for (int i = 0; i < odim; ++i) {
        float sum = weights_.b_out[sz(i)];
        for (int j = 0; j < H; ++j) {
            sum += weights_.W_out[sz(i) * sz(H) + sz(j)] * state.h[sz(j)];
        }
        out_buf_[sz(i)] = sum;
    }

    // --- 6. Store out_buf_ for SGD ---
    last_out_buf_[sz(layer)] = out_buf_;

    // --- 7. Convert to PredictionResult: per depth, softmax + argsort + top-k ---
    PredictionResult result;
    result.expert_ids.resize(sz(max_k_));
    result.confidences.resize(sz(max_k_));

    for (int d = 0; d < max_k_; ++d) {
        // Slice of out_buf_ for depth d: [d*num_experts .. (d+1)*num_experts)
        const int offset = d * num_experts_;
        std::vector<float> logits(sz(num_experts_));
        for (int e = 0; e < num_experts_; ++e) {
            logits[sz(e)] = out_buf_[sz(offset + e)];
        }

        // Softmax (numerically stable)
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp   = 0.0f;
        for (auto& v : logits) {
            v = std::exp(v - max_logit);
            sum_exp += v;
        }
        if (sum_exp > 0.0f) {
            for (auto& v : logits) {
                v /= sum_exp;
            }
        }

        // Argsort descending: get indices sorted by probability
        std::vector<uint32_t> indices(sz(num_experts_));
        std::iota(indices.begin(), indices.end(), static_cast<uint32_t>(0));
        std::sort(indices.begin(), indices.end(),
                  [&](uint32_t a, uint32_t b_idx) {
                      return logits[a] > logits[b_idx];
                  });

        result.expert_ids[sz(d)]  = indices;
        result.confidences[sz(d)].resize(sz(num_experts_));
        for (int e = 0; e < num_experts_; ++e) {
            result.confidences[sz(d)][sz(e)] = logits[indices[sz(e)]];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// LstmOracle::online_update -- online SGD on W_out only
// ---------------------------------------------------------------------------

void LstmOracle::online_update(int layer, int depth,
                                const std::vector<uint32_t>& actual_experts)
{
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };

    ++tokens_since_update_;
    if (tokens_since_update_ < UPDATE_INTERVAL) {
        return;
    }
    tokens_since_update_ = 0;

    // Timer gate: abort if we've already used 4ms.
    auto t_start = std::chrono::high_resolution_clock::now();

    const int H    = hidden_dim_;
    // depth is 1-indexed; row slice in W_out is [(depth-1)*num_experts .. depth*num_experts)
    const int d    = depth - 1;
    const int row_start = d * num_experts_;

    // Re-compute softmax probabilities for this depth from last stored output.
    const std::vector<float>& last = last_out_buf_[sz(layer)];
    std::vector<float> probs(sz(num_experts_));
    float max_logit = -1e30f;
    for (int e = 0; e < num_experts_; ++e) {
        if (last[sz(row_start + e)] > max_logit) {
            max_logit = last[sz(row_start + e)];
        }
    }
    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts_; ++e) {
        probs[sz(e)] = std::exp(last[sz(row_start + e)] - max_logit);
        sum_exp += probs[sz(e)];
    }
    if (sum_exp > 0.0f) {
        for (auto& p : probs) {
            p /= sum_exp;
        }
    }

    // Build one-hot label for actual_experts.
    std::vector<float> label(sz(num_experts_), 0.0f);
    for (uint32_t eid : actual_experts) {
        if (static_cast<int>(eid) < num_experts_) {
            label[eid] += 1.0f;
        }
    }
    // Normalise label so it sums to 1 (multi-hot -> distribution).
    float label_sum = 0.0f;
    for (float v : label) {
        label_sum += v;
    }
    if (label_sum > 0.0f) {
        for (auto& v : label) {
            v /= label_sum;
        }
    }

    // grad[e] = probs[e] - label[e]  (cross-entropy gradient)
    std::vector<float> grad(sz(num_experts_));
    for (int e = 0; e < num_experts_; ++e) {
        grad[sz(e)] = probs[sz(e)] - label[sz(e)];
    }

    // dW_out[row, col] = grad[row] * h[col]  (outer product, only for this depth slice)
    const LstmLayerState& state = layer_states_[sz(layer)];
    // Compute gradient norm.
    float norm_sq = 0.0f;
    for (int e = 0; e < num_experts_; ++e) {
        for (int j = 0; j < H; ++j) {
            float g = grad[sz(e)] * state.h[sz(j)];
            norm_sq += g * g;
        }
    }
    float norm  = std::sqrt(norm_sq);
    float scale = (norm > 1.0f) ? (1.0f / norm) : 1.0f;
    constexpr float LR = 1e-4f;

    // Timer check before update.
    auto t_now = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_now - t_start).count();
    if (elapsed_us > 4000) {
        return;  // budget exceeded, skip update
    }

    // Apply update.
    for (int e = 0; e < num_experts_; ++e) {
        int row = row_start + e;
        for (int j = 0; j < H; ++j) {
            float g = scale * grad[sz(e)] * state.h[sz(j)];
            weights_.W_out[sz(row) * sz(H) + sz(j)] -= LR * g;
        }
    }
}

// ---------------------------------------------------------------------------
// LstmOracle: state management
// ---------------------------------------------------------------------------

void LstmOracle::reset_layer_state(int layer) {
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };
    layer_states_[sz(layer)].reset(hidden_dim_);
}

void LstmOracle::reset_all_states() {
    for (auto& s : layer_states_) {
        s.reset(hidden_dim_);
    }
}

// ---------------------------------------------------------------------------
// LstmOracle: weight persistence
// ---------------------------------------------------------------------------

void LstmOracle::save_weights(const std::string& path) const {
    weights_.save(path);
}

bool LstmOracle::load_weights(const std::string& path) {
    LstmWeights tmp = weights_;  // keep current weights on failure
    if (!tmp.load(path)) {
        return false;
    }
    // Check dimensions match this oracle's configuration.
    if (tmp.input_dim   != weights_.input_dim   ||
        tmp.hidden_dim  != weights_.hidden_dim  ||
        tmp.output_dim  != weights_.output_dim  ||
        tmp.num_experts != weights_.num_experts ||
        tmp.max_k       != weights_.max_k) {
        return false;
    }
    weights_ = std::move(tmp);
    return true;
}

// ---------------------------------------------------------------------------
// LstmOracle::ram_bytes
// ---------------------------------------------------------------------------

size_t LstmOracle::ram_bytes() const {
    // Weight matrices
    size_t bytes = 0;
    bytes += weights_.W.size()      * sizeof(float);
    bytes += weights_.b.size()      * sizeof(float);
    bytes += weights_.W_out.size()  * sizeof(float);
    bytes += weights_.b_out.size()  * sizeof(float);
    bytes += weights_.W_proj.size() * sizeof(float);
    bytes += weights_.b_proj.size() * sizeof(float);

    // Per-layer states
    for (const auto& s : layer_states_) {
        bytes += s.h.size() * sizeof(float);
        bytes += s.c.size() * sizeof(float);
    }

    // Scratch buffers
    bytes += gate_buf_.size()    * sizeof(float);
    bytes += input_buf_.size()   * sizeof(float);
    bytes += proj_buf_.size()    * sizeof(float);
    bytes += out_buf_.size()     * sizeof(float);

    // Per-layer last output
    for (const auto& v : last_out_buf_) {
        bytes += v.size() * sizeof(float);
    }

    return bytes;
}

}  // namespace nos
