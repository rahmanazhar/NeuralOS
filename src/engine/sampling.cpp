/// @file sampling.cpp
/// @brief Token sampling transforms applied in order: repetition, temperature,
///        top-k, top-p, min-p, then categorical or greedy sampling.

#include "engine/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nos {

void apply_repetition_penalty(float* logits, int vocab_size,
                              const std::vector<int>& context_ids, float penalty) {
    if (penalty == 1.0f) return;
    for (int id : context_ids) {
        if (id < 0 || id >= vocab_size) continue;
        if (logits[id] > 0.0f) {
            logits[id] /= penalty;
        } else {
            logits[id] *= penalty;
        }
    }
}

void apply_temperature(float* logits, int vocab_size, float temp) {
    if (temp <= 0.0f) return;  // greedy handled separately
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }
}

void apply_top_k(float* logits, int vocab_size, int k) {
    if (k <= 0 || k >= vocab_size) return;

    // Find k-th largest value
    std::vector<float> sorted_logits(logits, logits + vocab_size);
    std::nth_element(sorted_logits.begin(),
                     sorted_logits.begin() + static_cast<ptrdiff_t>(k - 1),
                     sorted_logits.end(), std::greater<float>());
    float threshold = sorted_logits[static_cast<size_t>(k - 1)];

    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

void apply_top_p(float* logits, int vocab_size, float p) {
    if (p >= 1.0f) return;

    // Softmax
    float max_logit = *std::max_element(logits, logits + vocab_size);
    std::vector<std::pair<float, int>> probs(static_cast<size_t>(vocab_size));
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float prob = std::exp(logits[i] - max_logit);
        probs[static_cast<size_t>(i)] = {prob, i};
        sum += prob;
    }
    for (auto& pr : probs) {
        pr.first /= sum;
    }

    // Sort by probability descending
    std::sort(probs.begin(), probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Find cutoff
    float cumulative = 0.0f;
    size_t cutoff = probs.size();
    for (size_t i = 0; i < probs.size(); i++) {
        cumulative += probs[i].first;
        if (cumulative >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // Mark tokens below cutoff as -inf
    for (size_t i = cutoff; i < probs.size(); i++) {
        logits[probs[i].second] = -std::numeric_limits<float>::infinity();
    }
}

void apply_min_p(float* logits, int vocab_size, float min_p_val) {
    if (min_p_val <= 0.0f) return;

    float max_logit = *std::max_element(logits, logits + vocab_size);
    float threshold = max_logit + std::log(min_p_val);

    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

int greedy_sample(const float* logits, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

int sample_token(const float* logits, int vocab_size, std::mt19937& rng) {
    // Softmax with numerical stability
    float max_logit = *std::max_element(logits, logits + vocab_size);

    std::vector<float> probs(static_cast<size_t>(vocab_size));
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float val = logits[i];
        if (val == -std::numeric_limits<float>::infinity()) {
            probs[static_cast<size_t>(i)] = 0.0f;
        } else {
            probs[static_cast<size_t>(i)] = std::exp(val - max_logit);
        }
        sum += probs[static_cast<size_t>(i)];
    }

    if (sum <= 0.0f) return 0;  // fallback

    // Categorical sample
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float cumulative = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumulative += probs[static_cast<size_t>(i)];
        if (cumulative >= r) return i;
    }
    return vocab_size - 1;  // rounding edge case
}

int sample(float* logits, int vocab_size, const SamplingParams& params,
           const std::vector<int>& context, std::mt19937& rng) {
    if (params.temperature == 0.0f) {
        return greedy_sample(logits, vocab_size);
    }

    // Apply transforms in order
    apply_repetition_penalty(logits, vocab_size, context, params.repetition_penalty);
    apply_temperature(logits, vocab_size, params.temperature);
    apply_top_k(logits, vocab_size, params.top_k);
    apply_top_p(logits, vocab_size, params.top_p);
    apply_min_p(logits, vocab_size, params.min_p);

    return sample_token(logits, vocab_size, rng);
}

}  // namespace nos
