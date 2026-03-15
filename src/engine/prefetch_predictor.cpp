#include "prefetch_predictor.h"

#include <algorithm>
#include <cstddef>
#include <numeric>

namespace nos {

// ── rank_weighted_precision ───────────────────────────────────────────────────

float rank_weighted_precision(const std::vector<ExpertId>& predicted,
                               const std::vector<ExpertId>& actual) {
    if (predicted.empty() || actual.empty()) {
        return 0.0f;
    }

    // Build lookup set from actual
    // Use unordered_map for O(1) lookup
    std::unordered_map<ExpertId, bool> actual_set;
    actual_set.reserve(actual.size());
    for (ExpertId id : actual) {
        actual_set[id] = true;
    }

    float score    = 0.0f;
    float max_score = 0.0f;

    for (std::size_t i = 0; i < predicted.size(); ++i) {
        float weight = 1.0f / static_cast<float>(i + 1);
        max_score   += weight;
        if (actual_set.count(predicted[i]) > 0) {
            score += weight;
        }
    }

    if (max_score <= 0.0f) {
        return 0.0f;
    }
    return score / max_score;
}

// ── make_ngram_key ────────────────────────────────────────────────────────────

ContextKey make_ngram_key(const std::vector<ExpertId>& history, int n) {
    // Pack last (n-1) entries into uint64, 8 bits per entry
    int context_len = n - 1;
    if (context_len <= 0) {
        return 0ULL;
    }

    int history_size = static_cast<int>(history.size());
    int start        = history_size - context_len;
    if (start < 0) {
        start = 0;
    }

    ContextKey key = 0ULL;
    for (int i = start; i < history_size; ++i) {
        key = (key << 8) | (static_cast<ContextKey>(history[static_cast<std::size_t>(i)]) & 0xFFULL);
    }
    return key;
}

// ── NGramTable ────────────────────────────────────────────────────────────────

void NGramTable::observe(ContextKey ctx, ExpertId next) {
    // LRU eviction: if ctx is new and we're at capacity, evict oldest
    if (counts.find(ctx) == counts.end() &&
        counts.size() >= static_cast<std::size_t>(MAX_ENTRIES)) {
        // Evict oldest entry
        if (!insertion_order.empty()) {
            ContextKey oldest = insertion_order.front();
            insertion_order.pop_front();
            counts.erase(oldest);
        }
    }

    // Track insertion order for new entries
    if (counts.find(ctx) == counts.end()) {
        insertion_order.push_back(ctx);
    }

    counts[ctx][next]++;
}

std::vector<ExpertId> NGramTable::predict(ContextKey ctx, int top_k) const {
    auto it = counts.find(ctx);
    if (it == counts.end()) {
        return {};
    }

    const auto& successors = it->second;
    // Collect all (expert, count) pairs
    std::vector<std::pair<uint32_t, ExpertId>> sorted_experts;
    sorted_experts.reserve(successors.size());
    for (const auto& [expert_id, cnt] : successors) {
        sorted_experts.emplace_back(cnt, expert_id);
    }

    // Sort descending by count
    std::sort(sorted_experts.begin(), sorted_experts.end(),
              [](const std::pair<uint32_t, ExpertId>& a,
                 const std::pair<uint32_t, ExpertId>& b) {
                  return a.first > b.first;
              });

    std::vector<ExpertId> result;
    result.reserve(static_cast<std::size_t>(top_k));
    int limit = std::min(top_k, static_cast<int>(sorted_experts.size()));
    for (int i = 0; i < limit; ++i) {
        result.push_back(sorted_experts[static_cast<std::size_t>(i)].second);
    }
    return result;
}

// ── NGramPredictor ────────────────────────────────────────────────────────────

NGramPredictor::NGramPredictor(int n, int n_layers)
    : n_(n),
      per_layer_(static_cast<std::size_t>(n_layers)),
      history_(static_cast<std::size_t>(n_layers)) {}

void NGramPredictor::observe(int layer, const std::vector<ExpertId>& chosen) {
    std::size_t idx = static_cast<std::size_t>(layer);
    ContextKey ctx  = make_ngram_key(history_[idx], n_);
    for (ExpertId expert : chosen) {
        per_layer_[idx].observe(ctx, expert);
    }
    // Append all chosen experts to history
    for (ExpertId expert : chosen) {
        history_[idx].push_back(expert);
    }
}

std::vector<ExpertId> NGramPredictor::predict(int layer, int top_k) const {
    std::size_t idx = static_cast<std::size_t>(layer);
    ContextKey ctx  = make_ngram_key(history_[idx], n_);
    return per_layer_[idx].predict(ctx, top_k);
}

int NGramPredictor::gram_size() const {
    return n_;
}

// ── MarkovPredictor ───────────────────────────────────────────────────────────

MarkovPredictor::MarkovPredictor(int n_layers)
    : tables_(static_cast<std::size_t>(n_layers)),
      last_chosen_(static_cast<std::size_t>(n_layers), static_cast<ExpertId>(0)) {}

void MarkovPredictor::observe(int layer, const std::vector<ExpertId>& chosen) {
    if (chosen.empty()) {
        return;
    }
    std::size_t idx = static_cast<std::size_t>(layer);
    ExpertId    prev = last_chosen_[idx];

    // Record transition from prev to each chosen expert
    for (ExpertId expert : chosen) {
        tables_[idx][prev][expert]++;
    }

    // Update last representative as the first expert in chosen
    last_chosen_[idx] = chosen[0];
}

std::vector<ExpertId> MarkovPredictor::predict(int layer, int top_k) const {
    std::size_t idx  = static_cast<std::size_t>(layer);
    ExpertId    last = last_chosen_[idx];

    auto layer_it = tables_[idx].find(last);
    if (layer_it == tables_[idx].end()) {
        return {};
    }

    const auto& successors = layer_it->second;
    std::vector<std::pair<uint32_t, ExpertId>> sorted_experts;
    sorted_experts.reserve(successors.size());
    for (const auto& [expert_id, cnt] : successors) {
        sorted_experts.emplace_back(cnt, expert_id);
    }

    std::sort(sorted_experts.begin(), sorted_experts.end(),
              [](const std::pair<uint32_t, ExpertId>& a,
                 const std::pair<uint32_t, ExpertId>& b) {
                  return a.first > b.first;
              });

    std::vector<ExpertId> result;
    result.reserve(static_cast<std::size_t>(top_k));
    int limit = std::min(top_k, static_cast<int>(sorted_experts.size()));
    for (int i = 0; i < limit; ++i) {
        result.push_back(sorted_experts[static_cast<std::size_t>(i)].second);
    }
    return result;
}

// ── RepeatLastPredictor ───────────────────────────────────────────────────────

RepeatLastPredictor::RepeatLastPredictor(int n_layers)
    : last_experts_(static_cast<std::size_t>(n_layers)) {}

void RepeatLastPredictor::observe(int layer, const std::vector<ExpertId>& chosen) {
    last_experts_[static_cast<std::size_t>(layer)] = chosen;
}

std::vector<ExpertId> RepeatLastPredictor::predict(int layer, int top_k) const {
    std::size_t idx                    = static_cast<std::size_t>(layer);
    const std::vector<ExpertId>& last  = last_experts_[idx];

    if (last.empty()) {
        return {};
    }

    int limit = std::min(top_k, static_cast<int>(last.size()));
    return std::vector<ExpertId>(last.begin(),
                                  last.begin() + limit);
}

// ── MispredictionTracker ──────────────────────────────────────────────────────

// DepthStats

void MispredictionTracker::DepthStats::record(bool hit) {
    window[static_cast<std::size_t>(write_pos % WINDOW_SIZE)] = hit;
    write_pos++;
    total_recorded++;
}

double MispredictionTracker::DepthStats::hit_rate() const {
    if (total_recorded < WARMUP_TOKENS) {
        return -1.0;
    }
    int filled = std::min(total_recorded, WINDOW_SIZE);
    int hits   = 0;
    for (int i = 0; i < filled; ++i) {
        if (window[static_cast<std::size_t>(i)]) {
            hits++;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(filled);
}

// MispredictionTracker

MispredictionTracker::MispredictionTracker(int max_k, int n_layers)
    : max_k_(max_k),
      stats_(static_cast<std::size_t>(n_layers),
             std::vector<DepthStats>(static_cast<std::size_t>(max_k))),
      effective_k_(static_cast<std::size_t>(n_layers), max_k) {}

void MispredictionTracker::record_hit(int layer, int depth) {
    // depth is 1-indexed; stats_ is 0-indexed
    int d = depth - 1;
    if (d < 0 || d >= max_k_) {
        return;
    }
    stats_[static_cast<std::size_t>(layer)][static_cast<std::size_t>(d)].record(true);
}

void MispredictionTracker::record_miss(int layer, int depth) {
    int d = depth - 1;
    if (d < 0 || d >= max_k_) {
        return;
    }
    stats_[static_cast<std::size_t>(layer)][static_cast<std::size_t>(d)].record(false);
}

void MispredictionTracker::update_k(int layer) {
    std::size_t lidx    = static_cast<std::size_t>(layer);
    int         best_d  = 0;

    // Sweep depths max_k..1
    for (int d = max_k_; d >= 1; --d) {
        double rate = stats_[lidx][static_cast<std::size_t>(d - 1)].hit_rate();
        // -1.0 means warmup not complete — treat as passing
        if (rate < 0.0 || rate >= 0.70) {
            best_d = d;
            break;
        }
    }

    if (best_d == 0) {
        // No depth meets gate — set to minimum and potentially trigger cooldown
        effective_k_[lidx] = 1;
        // K=1 also failed: trigger cooldown
        start_cooldown();
    } else {
        effective_k_[lidx] = best_d;
    }
}

int MispredictionTracker::effective_k(int layer) const {
    return effective_k_[static_cast<std::size_t>(layer)];
}

void MispredictionTracker::tick_cooldown() {
    if (cooldown_remaining_ > 0) {
        cooldown_remaining_--;
    }
}

void MispredictionTracker::start_cooldown(int n) {
    cooldown_remaining_ = n;
}

bool MispredictionTracker::in_cooldown() const {
    return cooldown_remaining_ > 0;
}

void MispredictionTracker::reset_window(int layer) {
    std::size_t lidx = static_cast<std::size_t>(layer);
    for (auto& ds : stats_[lidx]) {
        ds = DepthStats{};
    }
}

}  // namespace nos
