#pragma once

/// @file sticky_router.h
/// @brief Sticky routing with adaptive lambda and stickiness window.
///
/// Wraps Router as a decorator, biasing routing decisions toward reusing
/// current experts via an additive lambda bonus. Lambda auto-tunes via
/// dual feedback: I/O pressure (cache miss rate) and sliding-window
/// perplexity delta.

#include "engine/router.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nos {

class StickyRouter {
public:
    struct Config {
        float lambda_override = -1.0f;   ///< <0 means auto; >=0 forces fixed lambda
        float w_io = 0.5f;              ///< I/O pressure weight in dual feedback
        float w_ppl = 0.5f;             ///< Perplexity delta weight
        int max_window = 128;            ///< Max stickiness window (tokens)
        float ema_alpha = 0.1f;          ///< EMA smoothing factor
        float ppl_threshold = 2.0f;      ///< Cross-entropy delta threshold for max ppl signal
    };

    /// Per-token trace entry for --trace-routing analysis.
    struct TraceEntry {
        int token_pos = 0;
        uint32_t layer_id = 0;
        std::vector<uint32_t> candidate_experts;
        std::vector<float> raw_scores;
        std::vector<float> adjusted_scores;
        float lambda = 0.0f;
        float io_pressure = 0.0f;
        float ppl_delta = 0.0f;
        bool switched = false;
        std::string reason;
    };

    StickyRouter();
    explicit StickyRouter(Config config);

    /// Initialize per-layer state for a model.
    void init(int num_layers);

    /// Route with stickiness applied. shift_detected overrides stickiness.
    RouterResult route(const Router& base_router,
                       const float* hidden_state, int k,
                       uint32_t layer_id, int token_pos,
                       bool shift_detected);

    /// Update feedback signals (called once per token, not per layer).
    void update_io_pressure(float cache_miss_rate);
    void update_perplexity(float cross_entropy_loss);

    /// Access last trace for --trace-routing output.
    const TraceEntry& last_trace() const { return last_trace_; }
    bool has_trace() const { return has_trace_; }

    /// Reset state (new sequence).
    void reset();

    /// Aggregate metrics.
    struct AggregateMetrics {
        uint64_t total_routing_decisions = 0;
        uint64_t total_switches = 0;
        float switch_rate = 0.0f;
        float avg_window_length = 0.0f;
    };
    AggregateMetrics aggregate_metrics() const;

    const Config& config() const { return config_; }

private:
    struct LayerState {
        std::vector<uint32_t> current_experts;
        int window_count = 0;
    };

    float compute_adaptive_lambda() const;

    Config config_;
    std::vector<LayerState> layer_states_;
    TraceEntry last_trace_;
    bool has_trace_ = false;

    // Feedback loop EMA state
    float io_pressure_ema_ = 0.0f;
    float ppl_ema_ = 0.0f;
    float ppl_ema_prev_ = 0.0f;

    // Aggregate metrics
    uint64_t total_decisions_ = 0;
    uint64_t total_switches_ = 0;
    uint64_t total_window_lengths_ = 0;
    uint64_t total_windows_ = 0;
};

}  // namespace nos
