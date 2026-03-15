#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace nos {

// ── Basic types ──────────────────────────────────────────────────────────────

using ExpertId   = uint32_t;
using ContextKey = uint64_t;

// ── rank_weighted_precision ───────────────────────────────────────────────────
// Harmonic rank-weighted precision: weight(i) = 1/(i+1)
// predicted: ranked predictions (index 0 = best), actual: ground truth top-k.
float rank_weighted_precision(const std::vector<ExpertId>& predicted,
                               const std::vector<ExpertId>& actual);

// ── make_ngram_key ────────────────────────────────────────────────────────────
// Packs last (n-1) entries from history into a uint64 context key.
// Uses 8 bits per ID (masks to low byte; approximate for models > 256 experts).
ContextKey make_ngram_key(const std::vector<ExpertId>& history, int n);

// ── NGramTable ────────────────────────────────────────────────────────────────
// Internal lookup table used by NGramPredictor.
struct NGramTable {
    static constexpr int MAX_ENTRIES = 50000;

    std::unordered_map<ContextKey,
        std::unordered_map<ExpertId, uint32_t>> counts;
    std::deque<ContextKey> insertion_order;  // LRU eviction tracking

    void observe(ContextKey ctx, ExpertId next);
    std::vector<ExpertId> predict(ContextKey ctx, int top_k) const;
};

// ── NGramPredictor ────────────────────────────────────────────────────────────
// Observes per-layer expert routing history and predicts top-k experts using
// n-gram statistics (1-gram through 4-gram).
class NGramPredictor {
public:
    // n: gram size 1..4. n_layers: number of transformer layers.
    explicit NGramPredictor(int n, int n_layers);

    // Record that 'chosen' experts were selected at this layer/token.
    void observe(int layer, const std::vector<ExpertId>& chosen);

    // Return top_k predicted expert IDs for next token at this layer.
    std::vector<ExpertId> predict(int layer, int top_k) const;

    int gram_size() const;

private:
    int n_;
    std::vector<NGramTable>               per_layer_;   // one per layer
    std::vector<std::vector<ExpertId>>    history_;     // rolling per-layer history
};

// ── MarkovPredictor ───────────────────────────────────────────────────────────
// Observes expert transitions (last expert -> next expert) per layer and
// predicts the most likely next experts.
class MarkovPredictor {
public:
    explicit MarkovPredictor(int n_layers);

    void observe(int layer, const std::vector<ExpertId>& chosen);
    std::vector<ExpertId> predict(int layer, int top_k) const;

private:
    // Per-layer: maps last expert -> frequency of next experts
    std::vector<
        std::unordered_map<ExpertId,
            std::unordered_map<ExpertId, uint32_t>>> tables_;
    std::vector<ExpertId> last_chosen_;   // last observed representative expert per layer
    std::vector<ExpertId> predict_from_;  // source of the most recent transition (for predict)
};

// ── RepeatLastPredictor ───────────────────────────────────────────────────────
// Predicts that the same experts chosen at layer L will be chosen again next
// token (zero-order baseline).
class RepeatLastPredictor {
public:
    explicit RepeatLastPredictor(int n_layers);

    void observe(int layer, const std::vector<ExpertId>& chosen);
    std::vector<ExpertId> predict(int layer, int top_k) const;

private:
    std::vector<std::vector<ExpertId>> last_experts_;  // per layer
};

// ── MispredictionTracker ──────────────────────────────────────────────────────
// Implements MEMO-05: per-layer per-depth sliding window accuracy tracking,
// auto-tunes effective_k, and triggers a cooldown when K=1 also fails.
class MispredictionTracker {
public:
    // max_k: maximum lookahead depth (e.g. 10). n_layers: transformer layer count.
    explicit MispredictionTracker(int max_k, int n_layers);

    // Record that expert at (layer, depth) was/wasn't in the predicted set.
    void record_hit(int layer, int depth);
    void record_miss(int layer, int depth);

    // Auto-tune: recompute effective_k for layer using 200-token sliding window.
    // Sweeps depths max_k..1, finds largest d with hit_rate >= 0.70.
    // If no depth meets the gate, sets effective_k to 1 and may trigger cooldown.
    void update_k(int layer);

    // Current auto-tuned lookahead for this layer (always >= 1).
    int effective_k(int layer) const;

    // Called once per token to decrement cooldown counter.
    void tick_cooldown();

    // Start a cooldown of n tokens (default 50) — oracle switches to baseline.
    void start_cooldown(int n = 50);

    bool in_cooldown() const;

    // Reset evaluation window (call on reset_kv_cache).
    void reset_window(int layer);

private:
    struct DepthStats {
        static constexpr int WINDOW_SIZE   = 200;
        static constexpr int WARMUP_TOKENS = 50;

        std::array<bool, WINDOW_SIZE> window{};
        int write_pos      = 0;
        int total_recorded = 0;

        void record(bool hit);
        // Returns hit rate over the window; -1.0 if fewer than WARMUP_TOKENS recorded.
        double hit_rate() const;
    };

    int max_k_;
    std::vector<std::vector<DepthStats>> stats_;    // [layer][depth], depth 0-indexed = depth 1
    std::vector<int>                     effective_k_;  // per-layer, initialized to max_k
    int cooldown_remaining_ = 0;
};

}  // namespace nos
