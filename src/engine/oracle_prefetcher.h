#pragma once

/// @file oracle_prefetcher.h
/// @brief Oracle prefetcher: integrates baseline predictors and LSTM oracle
///        into a single component that dispatches VMM prefetch_expert() calls
///        with dynamic confidence gating, fallback logic, and stderr visibility.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nos {

class Vmm;
class MetricsCollector;

/// Accumulated prefetch statistics for BenchmarkReporter.
struct PrefetchStats {
    std::string mode = "none";       ///< "oracle", "ngram_N" (N=gram size), "markov",
                                     ///<   "repeat_last", "none"
    double rwp_oracle       = 0.0;   ///< rank-weighted precision of oracle over last session
    double rwp_best_baseline = 0.0;  ///< RWP of the best-performing baseline predictor
    int    effective_k      = 0;     ///< current auto-tuned max lookahead K across all layers
    uint64_t speculative_hits   = 0; ///< prefetched experts subsequently used
    uint64_t speculative_misses = 0; ///< prefetched experts evicted unused
};

/// Integrates baseline predictors (NGram/Markov/RepeatLast) and LSTM oracle
/// into a single prefetch dispatch component.
///
/// - Dynamic confidence threshold driven by io_pressure EMA [0.30, 0.90]
/// - MEMO-05 fallback: if oracle misses 70% accuracy gate at K=1, switches to
///   best baseline and emits a stderr warning
/// - After 50-token cooldown, re-evaluates oracle and re-enables if gate passes
/// - Net-negative detection: wasted_io_ratio > 0.5 AND throughput_delta < -5%
///
/// OraclePrefetcher is non-copyable. Always use by pointer or unique_ptr.
class OraclePrefetcher {
public:
    struct Config {
        int n_layers                  = 32;
        int num_experts               = 64;
        int max_k                     = 10;
        int lstm_hidden_dim           = 64;
        int lstm_proj_dim             = 32;
        int online_update_interval    = 32;
        // Net-negative detection thresholds
        float net_neg_wasted_io_ratio  = 0.5f;
        float net_neg_throughput_delta = -0.05f;  ///< -5%
        int   cooldown_tokens          = 50;
    };

    /// @param config    OraclePrefetcher configuration.
    /// @param vmm       VMM instance for prefetch_expert() dispatch (non-owning, may be nullptr).
    /// @param metrics   Metrics collector for io_pressure timeline (non-owning, may be nullptr).
    explicit OraclePrefetcher(Config config, Vmm* vmm, MetricsCollector* metrics);
    ~OraclePrefetcher();

    OraclePrefetcher(const OraclePrefetcher&)            = delete;
    OraclePrefetcher& operator=(const OraclePrefetcher&) = delete;

    /// Called at each transformer layer L after routing.
    /// Feeds observation to all predictors and oracle.
    /// Dispatches VMM prefetch_expert() for predictions above confidence threshold.
    ///
    /// @param layer          Transformer layer index [0..n_layers).
    /// @param chosen_experts Expert IDs selected by the router this step.
    /// @param router_logits  num_experts floats (raw MoE gating scores).
    /// @param hidden_state   Engine hidden state (any size; used as LSTM input proxy).
    /// @param hidden_state_dim Size of hidden_state.
    /// @param top_k          Number of experts per layer (router top-k).
    void predict_and_dispatch(int layer,
                              const std::vector<uint32_t>& chosen_experts,
                              const float* router_logits,
                              const float* hidden_state,
                              int hidden_state_dim,
                              int top_k);

    /// Called once per token (after all layers done) to advance cooldown,
    /// record misprediction outcomes, and update throughput baseline.
    ///
    /// @param tok_per_sec_last_100  Approximate tokens/sec over last 100 tokens.
    void tick(double tok_per_sec_last_100);

    /// Reset state for a new sequence (call on reset_kv_cache).
    void reset();

    /// Snapshot of accumulated prefetch statistics.
    PrefetchStats stats() const;

    /// Enable/disable dispatching (default: enabled when constructed).
    void set_enabled(bool enabled);
    bool enabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nos
