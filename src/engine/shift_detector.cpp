/// @file shift_detector.cpp
/// @brief Multi-signal semantic shift detector implementation.

#include "engine/shift_detector.h"

#include <algorithm>
#include <cmath>

namespace nos {

// Normalization ceilings for delta signals
static constexpr float MAX_ENTROPY_DELTA = 2.0f;
static constexpr float MAX_VARIANCE_DELTA = 5.0f;

ShiftDetector::ShiftDetector() : config_() {}
ShiftDetector::ShiftDetector(Config config) : config_(config) {}

ShiftDetector::ShiftResult ShiftDetector::evaluate(
        float attn_entropy, float cosine_sim, float router_logit_variance) {

    ShiftResult result{};

    if (first_token_) {
        prev_entropy_ = attn_entropy;
        prev_variance_ = router_logit_variance;
        first_token_ = false;
        tokens_since_switch_ = 0;
        result.confidence = 0.0f;
        result.detected = false;
        result.entropy_delta = 0.0f;
        result.cosine_sim = cosine_sim;
        result.variance_delta = 0.0f;
        return result;
    }

    // Entropy signal: large change in attention entropy -> shift
    float entropy_raw = std::abs(attn_entropy - prev_entropy_);
    float entropy_signal = std::clamp(entropy_raw / MAX_ENTROPY_DELTA, 0.0f, 1.0f);

    // Cosine signal: low similarity -> shift (1 - cos_sim)
    float cosine_signal = 1.0f - std::clamp(cosine_sim, 0.0f, 1.0f);

    // Variance signal: large change in router logit variance -> shift
    float variance_raw = std::abs(router_logit_variance - prev_variance_);
    float variance_signal = std::clamp(variance_raw / MAX_VARIANCE_DELTA, 0.0f, 1.0f);

    // Weighted blend (weights can be 0 for ablation)
    float total_weight = config_.entropy_weight + config_.cosine_weight
                       + config_.variance_weight;
    float confidence = 0.0f;
    if (total_weight > 0.0f) {
        confidence = (config_.entropy_weight * entropy_signal
                    + config_.cosine_weight * cosine_signal
                    + config_.variance_weight * variance_signal) / total_weight;
    }

    result.confidence = confidence;
    result.entropy_delta = entropy_signal;
    result.cosine_sim = cosine_sim;
    result.variance_delta = variance_signal;

    // Cooldown: suppress detection for current_cooldown_ tokens after a switch
    if (tokens_since_switch_ < current_cooldown_) {
        result.detected = false;
    } else {
        result.detected = confidence > config_.threshold;
    }

    // If detected, set cooldown and reset counter
    if (result.detected) {
        current_cooldown_ = cooldown_tokens(confidence);
        tokens_since_switch_ = 0;
    } else {
        tokens_since_switch_++;
    }

    prev_entropy_ = attn_entropy;
    prev_variance_ = router_logit_variance;

    return result;
}

int ShiftDetector::cooldown_tokens(float confidence) const {
    // High confidence -> short cooldown (min), low -> long cooldown (max)
    float t = std::clamp(confidence, 0.0f, 1.0f);
    float cd = static_cast<float>(config_.cooldown_max)
             - t * static_cast<float>(config_.cooldown_max - config_.cooldown_min);
    return static_cast<int>(cd);
}

void ShiftDetector::reset() {
    prev_entropy_ = 0.0f;
    prev_variance_ = 0.0f;
    tokens_since_switch_ = 0;
    current_cooldown_ = 0;
    first_token_ = true;
}

float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

}  // namespace nos
