/// @file sticky_router.cpp
/// @brief Sticky router with dual feedback loop implementation.

#include "engine/sticky_router.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nos {

StickyRouter::StickyRouter() : config_() {}
StickyRouter::StickyRouter(Config config) : config_(config) {}

void StickyRouter::init(int num_layers) {
    layer_states_.resize(static_cast<size_t>(num_layers));
    for (auto& ls : layer_states_) {
        ls.current_experts.clear();
        ls.window_count = 0;
    }
    has_trace_ = false;
}

RouterResult StickyRouter::route(const Router& base_router,
                                  const float* hidden_state, int k,
                                  uint32_t layer_id, int token_pos,
                                  bool shift_detected) {
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };

    // Get raw routing result from base router
    RouterResult base_result = base_router.route(hidden_state, k);
    int num_experts = base_router.num_experts();

    auto& ls = layer_states_[static_cast<size_t>(layer_id)];

    // Compute lambda
    float lambda = (config_.lambda_override >= 0.0f)
        ? config_.lambda_override
        : compute_adaptive_lambda();

    // Determine if we should apply stickiness
    bool skip_sticky = false;
    std::string reason = "sticky";

    if (shift_detected) {
        skip_sticky = true;
        reason = "shift_detected";
    } else if (ls.window_count >= config_.max_window) {
        skip_sticky = true;
        reason = "window_expired";
    } else if (lambda < 1e-6f) {
        skip_sticky = true;
        reason = "lambda_zero";
    } else if (ls.current_experts.empty()) {
        skip_sticky = true;
        reason = "first_routing";
    }

    RouterResult result;

    if (skip_sticky) {
        // Fresh routing — use base result as-is
        result = base_result;
    } else {
        // Apply stickiness bonus to current experts
        std::vector<float> adjusted(sz(num_experts));
        for (int e = 0; e < num_experts; e++) {
            adjusted[sz(e)] = base_result.raw_scores[sz(e)];
        }

        // Add lambda bonus for experts that are currently in use
        for (uint32_t ce : ls.current_experts) {
            if (static_cast<int>(ce) < num_experts) {
                adjusted[static_cast<size_t>(ce)] += lambda;
            }
        }

        // Re-select top-k from adjusted scores
        std::vector<size_t> indices(sz(num_experts));
        std::iota(indices.begin(), indices.end(), size_t{0});
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&adjusted](size_t a, size_t b) {
                              return adjusted[a] > adjusted[b];
                          });

        result.expert_ids.resize(sz(k));
        result.gates.resize(sz(k));
        result.raw_scores = base_result.raw_scores;

        // Softmax renormalization on adjusted top-k scores
        float max_adj = adjusted[indices[0]];
        float sum_exp = 0.0f;
        for (size_t i = 0; i < sz(k); i++) {
            result.gates[i] = std::exp(adjusted[indices[i]] - max_adj);
            sum_exp += result.gates[i];
        }
        float inv_sum = 1.0f / sum_exp;
        for (size_t i = 0; i < sz(k); i++) {
            result.expert_ids[i] = static_cast<uint32_t>(indices[i]);
            result.gates[i] *= inv_sum;
        }

        // Populate trace with adjusted scores
        last_trace_.adjusted_scores.resize(sz(num_experts));
        for (int e = 0; e < num_experts; e++) {
            last_trace_.adjusted_scores[sz(e)] = adjusted[sz(e)];
        }
    }

    // Determine if experts changed
    bool switched = false;
    if (ls.current_experts.size() != result.expert_ids.size()) {
        switched = true;
    } else {
        std::vector<uint32_t> sorted_curr = ls.current_experts;
        std::vector<uint32_t> sorted_new = result.expert_ids;
        std::sort(sorted_curr.begin(), sorted_curr.end());
        std::sort(sorted_new.begin(), sorted_new.end());
        switched = (sorted_curr != sorted_new);
    }

    // Update layer state
    total_decisions_++;
    if (switched) {
        total_switches_++;
        if (ls.window_count > 0) {
            total_window_lengths_ += static_cast<uint64_t>(ls.window_count);
            total_windows_++;
        }
        ls.current_experts = result.expert_ids;
        ls.window_count = 0;
    } else {
        ls.window_count++;
    }

    // Populate trace entry
    last_trace_.token_pos = token_pos;
    last_trace_.layer_id = layer_id;
    last_trace_.candidate_experts = result.expert_ids;
    last_trace_.raw_scores = base_result.raw_scores;
    if (skip_sticky) {
        last_trace_.adjusted_scores = base_result.raw_scores;
    }
    last_trace_.lambda = lambda;
    last_trace_.io_pressure = io_pressure_ema_;
    last_trace_.ppl_delta = std::abs(ppl_ema_ - ppl_ema_prev_);
    last_trace_.switched = switched;
    last_trace_.reason = reason;
    has_trace_ = true;

    return result;
}

void StickyRouter::update_io_pressure(float cache_miss_rate) {
    io_pressure_ema_ = config_.ema_alpha * cache_miss_rate
                     + (1.0f - config_.ema_alpha) * io_pressure_ema_;
}

void StickyRouter::update_perplexity(float cross_entropy_loss) {
    ppl_ema_prev_ = ppl_ema_;
    ppl_ema_ = config_.ema_alpha * cross_entropy_loss
             + (1.0f - config_.ema_alpha) * ppl_ema_;
}

float StickyRouter::compute_adaptive_lambda() const {
    // High I/O pressure -> more sticky (high lambda)
    float io_signal = std::clamp(io_pressure_ema_, 0.0f, 1.0f);

    // Stable perplexity -> more sticky (high signal)
    float ppl_delta = std::abs(ppl_ema_ - ppl_ema_prev_);
    float ppl_signal = 1.0f - std::clamp(ppl_delta / config_.ppl_threshold, 0.0f, 1.0f);

    float total_weight = config_.w_io + config_.w_ppl;
    if (total_weight < 1e-6f) return 0.0f;

    float lambda = (config_.w_io * io_signal + config_.w_ppl * ppl_signal) / total_weight;
    return std::clamp(lambda, 0.0f, 1.0f);
}

void StickyRouter::reset() {
    for (auto& ls : layer_states_) {
        ls.current_experts.clear();
        ls.window_count = 0;
    }
    io_pressure_ema_ = 0.0f;
    ppl_ema_ = 0.0f;
    ppl_ema_prev_ = 0.0f;
    total_decisions_ = 0;
    total_switches_ = 0;
    total_window_lengths_ = 0;
    total_windows_ = 0;
    has_trace_ = false;
}

StickyRouter::AggregateMetrics StickyRouter::aggregate_metrics() const {
    AggregateMetrics m;
    m.total_routing_decisions = total_decisions_;
    m.total_switches = total_switches_;
    m.switch_rate = (total_decisions_ > 0)
        ? static_cast<float>(total_switches_) / static_cast<float>(total_decisions_)
        : 0.0f;
    m.avg_window_length = (total_windows_ > 0)
        ? static_cast<float>(total_window_lengths_) / static_cast<float>(total_windows_)
        : 0.0f;
    return m;
}

}  // namespace nos
