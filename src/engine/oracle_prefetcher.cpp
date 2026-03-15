/// @file oracle_prefetcher.cpp
/// @brief OraclePrefetcher implementation: predictor selection, VMM dispatch,
///        dynamic confidence threshold, fallback/restore logic, stderr messages.

#include "engine/oracle_prefetcher.h"

#include "engine/metrics.h"
#include "engine/oracle_lstm.h"
#include "engine/prefetch_predictor.h"
#include "vmm/vmm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace nos {

// ── Constants ──────────────────────────────────────────────────────────────

static constexpr float  MIN_THRESHOLD      = 0.30f;
static constexpr float  EMA_ALPHA          = 0.3f;
static constexpr int    BASELINE_RWP_WINDOW = 200;
static constexpr int    K_UPDATE_INTERVAL   = 10;   // tokens between update_k calls

// ── Impl ────────────────────────────────────────────────────────────────────

struct OraclePrefetcher::Impl {
    Config config;
    Vmm*              vmm_     = nullptr;  // non-owning
    MetricsCollector* metrics_ = nullptr;  // non-owning

    // Baseline predictors (all running independently)
    NGramPredictor   ngram1_;
    NGramPredictor   ngram2_;
    NGramPredictor   ngram3_;
    NGramPredictor   ngram4_;
    MarkovPredictor  markov_;
    RepeatLastPredictor repeat_last_;

    // LSTM oracle
    LstmOracle lstm_oracle_;

    // Misprediction tracking / auto-tuned K
    MispredictionTracker mispred_tracker_;

    // State
    PrefetchStats stats_;
    std::string   active_predictor_ = "oracle";
    bool          oracle_enabled_   = true;
    bool          enabled_          = true;

    // Net-negative detection: running tok/s baseline (exponential average)
    double baseline_tok_per_sec_    = 0.0;
    bool   baseline_initialised_    = false;

    // Pending predictions issued at the PREVIOUS token for each layer.
    // pending_predictions_[layer] = expert IDs predicted for layer at the previous token.
    std::vector<std::vector<uint32_t>> pending_predictions_;

    // Per-layer LSTM result from last forward_step (depth=0 = depth 1, etc.)
    std::vector<LstmOracle::PredictionResult> last_lstm_result_;

    // EMA state for io_pressure threshold
    float io_pressure_ema_ = 0.0f;
    bool  ema_initialised_  = false;

    // Token counter
    uint64_t total_tokens_ = 0;

    // RWP sliding window for each predictor (raw per-token scores)
    struct RwpWindow {
        std::vector<float> values;  // circular buffer of size BASELINE_RWP_WINDOW
        int   write_pos = 0;
        int   count     = 0;

        void push(float v) {
            if (static_cast<int>(values.size()) < BASELINE_RWP_WINDOW) {
                values.resize(static_cast<size_t>(BASELINE_RWP_WINDOW), 0.0f);
            }
            values[static_cast<size_t>(write_pos)] = v;
            write_pos = (write_pos + 1) % BASELINE_RWP_WINDOW;
            if (count < BASELINE_RWP_WINDOW) count++;
        }

        double mean() const {
            if (count == 0) return 0.0;
            double s = 0.0;
            for (int i = 0; i < count; i++) {
                s += static_cast<double>(values[static_cast<size_t>(i)]);
            }
            return s / static_cast<double>(count);
        }
    };

    // One RWP window per predictor: ngram1-4, markov, repeat_last
    RwpWindow rwp_ngram1_, rwp_ngram2_, rwp_ngram3_, rwp_ngram4_;
    RwpWindow rwp_markov_, rwp_repeat_last_;
    // Also track oracle RWP
    RwpWindow rwp_oracle_;

    // Transition-once flags to prevent stderr spam
    bool fallback_message_sent_  = false;
    bool restore_message_sent_   = false;

    // ── Constructor ───────────────────────────────────────────────────────

    Impl(Config cfg, Vmm* vmm, MetricsCollector* metrics)
        : config(cfg)
        , vmm_(vmm)
        , metrics_(metrics)
        , ngram1_(1, cfg.n_layers)
        , ngram2_(2, cfg.n_layers)
        , ngram3_(3, cfg.n_layers)
        , ngram4_(4, cfg.n_layers)
        , markov_(cfg.n_layers)
        , repeat_last_(cfg.n_layers)
        , lstm_oracle_(cfg.num_experts, cfg.n_layers,
                       cfg.lstm_hidden_dim, cfg.max_k, cfg.lstm_proj_dim)
        , mispred_tracker_(cfg.max_k, cfg.n_layers)
        , pending_predictions_(static_cast<size_t>(cfg.n_layers))
        , last_lstm_result_(static_cast<size_t>(cfg.n_layers))
    {
        lstm_oracle_.init_weights();
        stats_.effective_k = cfg.max_k;
    }

    // ── Dynamic confidence threshold ─────────────────────────────────────

    float get_confidence_threshold() {
        if (metrics_) {
            auto tl = metrics_->get_timeline("io_pressure");
            // Use last 5 points
            int n_points = static_cast<int>(tl.points.size());
            int start = std::max(0, n_points - 5);
            for (int i = start; i < n_points; i++) {
                float val = static_cast<float>(tl.points[static_cast<size_t>(i)].second);
                if (!ema_initialised_) {
                    io_pressure_ema_ = val;
                    ema_initialised_ = true;
                } else {
                    io_pressure_ema_ = EMA_ALPHA * val + (1.0f - EMA_ALPHA) * io_pressure_ema_;
                }
            }
        }
        return MIN_THRESHOLD + 0.60f * std::clamp(io_pressure_ema_, 0.0f, 1.0f);
    }

    // ── Best baseline selection ───────────────────────────────────────────

    // Returns the name of the best baseline and its average RWP.
    std::pair<std::string, double> best_baseline() const {
        struct Entry { std::string name; double rwp; };
        std::vector<Entry> candidates = {
            { "ngram_1",     rwp_ngram1_.mean()     },
            { "ngram_2",     rwp_ngram2_.mean()     },
            { "ngram_3",     rwp_ngram3_.mean()     },
            { "ngram_4",     rwp_ngram4_.mean()     },
            { "markov",      rwp_markov_.mean()     },
            { "repeat_last", rwp_repeat_last_.mean()},
        };
        auto best = std::max_element(candidates.begin(), candidates.end(),
            [](const Entry& a, const Entry& b) { return a.rwp < b.rwp; });
        return { best->name, best->rwp };
    }

    // ── Predictor predictions ─────────────────────────────────────────────

    // Get top_k expert predictions from the currently active baseline (by name).
    std::vector<uint32_t> baseline_predict(const std::string& name, int layer, int k) const {
        if (name == "ngram_1")     return ngram1_.predict(layer, k);
        if (name == "ngram_2")     return ngram2_.predict(layer, k);
        if (name == "ngram_3")     return ngram3_.predict(layer, k);
        if (name == "ngram_4")     return ngram4_.predict(layer, k);
        if (name == "markov")      return markov_.predict(layer, k);
        if (name == "repeat_last") return repeat_last_.predict(layer, k);
        return {};
    }

    // ── Misprediction accounting (deferred: verify previous token's prediction) ──

    void account_mispredictions(int layer, const std::vector<uint32_t>& chosen) {
        if (pending_predictions_[static_cast<size_t>(layer)].empty()) return;

        // The pending_predictions_ for this layer were issued at depth=1 from
        // the previous token. Verify against actual chosen.
        const auto& pred = pending_predictions_[static_cast<size_t>(layer)];
        bool any_hit = false;
        for (uint32_t p : pred) {
            for (uint32_t c : chosen) {
                if (p == c) { any_hit = true; break; }
            }
            if (any_hit) break;
        }
        if (any_hit) {
            mispred_tracker_.record_hit(layer, 1);
            stats_.speculative_hits++;
        } else {
            mispred_tracker_.record_miss(layer, 1);
            stats_.speculative_misses++;
        }

        // RWP accounting: build predicted and actual vectors
        std::vector<uint32_t> actual_v(chosen.begin(), chosen.end());
        float rwp = rank_weighted_precision(pred, actual_v);

        // Record for oracle vs active predictor
        if (oracle_enabled_ && !mispred_tracker_.in_cooldown()) {
            rwp_oracle_.push(rwp);
        } else {
            // Push to whichever baseline was active
            if (active_predictor_ == "ngram_1")     rwp_ngram1_.push(rwp);
            else if (active_predictor_ == "ngram_2") rwp_ngram2_.push(rwp);
            else if (active_predictor_ == "ngram_3") rwp_ngram3_.push(rwp);
            else if (active_predictor_ == "ngram_4") rwp_ngram4_.push(rwp);
            else if (active_predictor_ == "markov")  rwp_markov_.push(rwp);
            else if (active_predictor_ == "repeat_last") rwp_repeat_last_.push(rwp);
        }
    }
};

// ── OraclePrefetcher public API ─────────────────────────────────────────────

OraclePrefetcher::OraclePrefetcher(Config config, Vmm* vmm, MetricsCollector* metrics)
    : impl_(std::make_unique<Impl>(config, vmm, metrics))
{}

OraclePrefetcher::~OraclePrefetcher() = default;

void OraclePrefetcher::set_enabled(bool enabled) {
    impl_->enabled_ = enabled;
}

bool OraclePrefetcher::enabled() const {
    return impl_->enabled_;
}

PrefetchStats OraclePrefetcher::stats() const {
    PrefetchStats s = impl_->stats_;
    s.mode = impl_->active_predictor_;
    if (!impl_->enabled_) s.mode = "none";

    // Aggregate effective_k: use minimum effective_k across all layers
    int min_k = impl_->config.max_k;
    for (int l = 0; l < impl_->config.n_layers; l++) {
        int ek = impl_->mispred_tracker_.effective_k(l);
        if (ek < min_k) min_k = ek;
    }
    s.effective_k = min_k;

    s.rwp_oracle         = impl_->rwp_oracle_.mean();
    s.rwp_best_baseline  = impl_->best_baseline().second;
    return s;
}

void OraclePrefetcher::predict_and_dispatch(int layer,
                                             const std::vector<uint32_t>& chosen_experts,
                                             const float* router_logits,
                                             const float* hidden_state,
                                             int hidden_state_dim,
                                             int top_k)
{
    if (!impl_->enabled_) return;

    int n_layers = impl_->config.n_layers;

    // Step 1: Feed observation to all baseline predictors
    impl_->ngram1_.observe(layer, chosen_experts);
    impl_->ngram2_.observe(layer, chosen_experts);
    impl_->ngram3_.observe(layer, chosen_experts);
    impl_->ngram4_.observe(layer, chosen_experts);
    impl_->markov_.observe(layer, chosen_experts);
    impl_->repeat_last_.observe(layer, chosen_experts);

    // Step 2: Feed observation to LSTM oracle (save result for dispatch)
    LstmOracle::PredictionResult lstm_result;
    if (router_logits != nullptr) {
        lstm_result = impl_->lstm_oracle_.forward_step(
            layer, router_logits, hidden_state, hidden_state_dim);
    }
    impl_->last_lstm_result_[static_cast<size_t>(layer)] = lstm_result;

    // Step 3: Misprediction accounting from previous token's predictions
    impl_->account_mispredictions(layer, chosen_experts);

    // Step 4: Determine active predictor
    bool use_oracle = impl_->oracle_enabled_ && !impl_->mispred_tracker_.in_cooldown();
    if (!use_oracle && impl_->active_predictor_ == "oracle") {
        // Switch to best baseline
        impl_->active_predictor_ = impl_->best_baseline().first;
    } else if (use_oracle) {
        impl_->active_predictor_ = "oracle";
    }

    // Step 5: Get confidence threshold
    float thresh = impl_->get_confidence_threshold();

    // Step 6: Dispatch prefetch calls for depths 1..effective_k
    int eff_k = impl_->mispred_tracker_.effective_k(layer);

    // Clear pending for this layer before recording new predictions
    impl_->pending_predictions_[static_cast<size_t>(layer)].clear();

    for (int depth = 1; depth <= eff_k; depth++) {
        int target_layer = layer + depth;
        if (target_layer >= n_layers) break;

        std::vector<uint32_t> predicted;
        std::vector<float>    confidences;

        if (use_oracle && !lstm_result.expert_ids.empty()) {
            int d_idx = depth - 1;
            if (d_idx < static_cast<int>(lstm_result.expert_ids.size())) {
                predicted   = lstm_result.expert_ids[static_cast<size_t>(d_idx)];
                if (d_idx < static_cast<int>(lstm_result.confidences.size())) {
                    confidences = lstm_result.confidences[static_cast<size_t>(d_idx)];
                }
            }
        } else {
            // Baseline: uniform confidence = 1.0 (always above threshold at 0.30)
            predicted = impl_->baseline_predict(impl_->active_predictor_, layer, top_k);
            confidences.assign(predicted.size(), 1.0f);
        }

        for (size_t pi = 0; pi < predicted.size(); pi++) {
            float conf = (pi < confidences.size()) ? confidences[pi] : 1.0f;
            if (conf >= thresh) {
                if (impl_->vmm_) {
                    impl_->vmm_->prefetch_expert(
                        static_cast<uint32_t>(target_layer), predicted[pi]);
                }
                // Record first-depth predictions for next-token misprediction accounting
                if (depth == 1) {
                    impl_->pending_predictions_[static_cast<size_t>(layer)]
                        .push_back(predicted[pi]);
                }
            }
        }
    }

    // Step 7: Periodic update_k (every K_UPDATE_INTERVAL tokens, on layer 0)
    if (layer == 0 && impl_->total_tokens_ > 0
            && impl_->total_tokens_ % static_cast<uint64_t>(K_UPDATE_INTERVAL) == 0) {
        for (int l = 0; l < n_layers; l++) {
            impl_->mispred_tracker_.update_k(l);
        }
    }
}

void OraclePrefetcher::tick(double tok_per_sec_last_100) {
    // Step 1: Advance cooldown — snapshot BEFORE tick to detect just-expired
    bool was_in_cooldown = impl_->mispred_tracker_.in_cooldown();
    impl_->mispred_tracker_.tick_cooldown();
    bool now_in_cooldown = impl_->mispred_tracker_.in_cooldown();

    // Detect cooldown just expired (was in cooldown, now not)
    bool just_released = was_in_cooldown && !now_in_cooldown;

    // Step 2: Net-negative detection (only when oracle is active)
    if (impl_->oracle_enabled_ && !impl_->mispred_tracker_.in_cooldown()
            && impl_->stats_.speculative_misses > 0) {
        uint64_t total_prefetches = impl_->stats_.speculative_hits
                                  + impl_->stats_.speculative_misses;
        float wasted_io_ratio = (total_prefetches > 0)
            ? static_cast<float>(impl_->stats_.speculative_misses)
              / static_cast<float>(total_prefetches)
            : 0.0f;

        double throughput_delta = 0.0;
        if (impl_->baseline_initialised_ && impl_->baseline_tok_per_sec_ > 0.0) {
            throughput_delta = (tok_per_sec_last_100 - impl_->baseline_tok_per_sec_)
                             / impl_->baseline_tok_per_sec_;
        }

        bool net_negative = (wasted_io_ratio > impl_->config.net_neg_wasted_io_ratio)
                         && (throughput_delta < static_cast<double>(
                                 impl_->config.net_neg_throughput_delta));

        if (net_negative) {
            impl_->oracle_enabled_    = false;
            impl_->fallback_message_sent_ = false;  // allow message below
            impl_->restore_message_sent_  = false;
            impl_->mispred_tracker_.start_cooldown(impl_->config.cooldown_tokens);
        }
    }

    // Step 3: Fallback message (once per transition into cooldown / oracle-off state)
    if (!impl_->oracle_enabled_ && !impl_->fallback_message_sent_) {
        auto [best_name, best_rwp] = impl_->best_baseline();
        // Find best effective_k across all layers for the message
        int best_k = 1;
        for (int l = 0; l < impl_->config.n_layers; l++) {
            int ek = impl_->mispred_tracker_.effective_k(l);
            if (ek > best_k) best_k = ek;
        }
        std::fprintf(stderr,
            "[neuralos] oracle accuracy below threshold, using %s predictor (K=%d)\n",
            best_name.c_str(), best_k);
        impl_->fallback_message_sent_ = true;
        impl_->active_predictor_ = best_name;
        impl_->stats_.mode = best_name;
    }

    // Step 4: Re-evaluate oracle after cooldown expires
    if (just_released) {
        // Force update_k for all layers and check if oracle should be re-enabled
        for (int l = 0; l < impl_->config.n_layers; l++) {
            impl_->mispred_tracker_.update_k(l);
        }
        // Re-enable oracle if at least one layer has effective_k > 1
        bool should_restore = false;
        for (int l = 0; l < impl_->config.n_layers; l++) {
            if (impl_->mispred_tracker_.effective_k(l) > 1) {
                should_restore = true;
                break;
            }
        }
        if (should_restore) {
            impl_->oracle_enabled_        = true;
            impl_->active_predictor_      = "oracle";
            impl_->stats_.mode            = "oracle";
            impl_->fallback_message_sent_ = false;
        }

        // Restore message (once per transition back to oracle)
        if (should_restore && !impl_->restore_message_sent_) {
            std::fprintf(stderr,
                "[neuralos] oracle restored, resuming speculative prefetch\n");
            impl_->restore_message_sent_ = true;
        }
    }

    // Step 5: Update throughput baseline (EMA with alpha=0.1 to be stable)
    if (tok_per_sec_last_100 > 0.0) {
        if (!impl_->baseline_initialised_) {
            impl_->baseline_tok_per_sec_ = tok_per_sec_last_100;
            impl_->baseline_initialised_ = true;
        } else {
            impl_->baseline_tok_per_sec_ =
                0.1 * tok_per_sec_last_100 + 0.9 * impl_->baseline_tok_per_sec_;
        }
    }

    // Step 6: Update stats summary
    impl_->stats_.mode        = impl_->active_predictor_;
    impl_->stats_.effective_k = impl_->mispred_tracker_.effective_k(0);

    impl_->total_tokens_++;
}

void OraclePrefetcher::reset() {
    int n_layers    = impl_->config.n_layers;
    int num_experts = impl_->config.num_experts;
    int max_k       = impl_->config.max_k;

    // Reset baseline predictors
    impl_->ngram1_      = NGramPredictor(1, n_layers);
    impl_->ngram2_      = NGramPredictor(2, n_layers);
    impl_->ngram3_      = NGramPredictor(3, n_layers);
    impl_->ngram4_      = NGramPredictor(4, n_layers);
    impl_->markov_      = MarkovPredictor(n_layers);
    impl_->repeat_last_ = RepeatLastPredictor(n_layers);

    // Reset oracle states
    impl_->lstm_oracle_.reset_all_states();

    // Reset misprediction tracker windows for all layers
    for (int l = 0; l < n_layers; l++) {
        impl_->mispred_tracker_.reset_window(l);
    }

    // Reset pending predictions
    for (auto& v : impl_->pending_predictions_) v.clear();

    // Reset LSTM results
    for (auto& r : impl_->last_lstm_result_) {
        r.expert_ids.clear();
        r.confidences.clear();
    }

    // Reset cooldown state
    // (MispredictionTracker doesn't expose a full reset, but reset_window + new tracker does)
    impl_->mispred_tracker_ = MispredictionTracker(max_k, n_layers);

    // Reset state flags
    impl_->oracle_enabled_        = true;
    impl_->active_predictor_      = "oracle";
    impl_->fallback_message_sent_ = false;
    impl_->restore_message_sent_  = false;
    impl_->total_tokens_          = 0;
    impl_->ema_initialised_       = false;
    impl_->io_pressure_ema_       = 0.0f;
    impl_->baseline_initialised_  = false;
    impl_->baseline_tok_per_sec_  = 0.0;

    // Reset stats counters (keep mode as oracle)
    impl_->stats_                    = PrefetchStats{};
    impl_->stats_.effective_k        = max_k;

    // Reset RWP windows
    impl_->rwp_ngram1_     = Impl::RwpWindow{};
    impl_->rwp_ngram2_     = Impl::RwpWindow{};
    impl_->rwp_ngram3_     = Impl::RwpWindow{};
    impl_->rwp_ngram4_     = Impl::RwpWindow{};
    impl_->rwp_markov_     = Impl::RwpWindow{};
    impl_->rwp_repeat_last_= Impl::RwpWindow{};
    impl_->rwp_oracle_     = Impl::RwpWindow{};

    (void)num_experts;  // used in constructor only
}

}  // namespace nos
