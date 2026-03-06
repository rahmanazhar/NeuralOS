#pragma once

/// @file shift_detector.h
/// @brief Multi-signal semantic shift detector for sticky routing override.
///
/// Fuses three signals (attention entropy delta, hidden state cosine similarity,
/// router logit variance delta) to detect topic changes that should override
/// sticky routing. Individual signals can be ablated by setting weight to 0.

#include <cstdint>

namespace nos {

class ShiftDetector {
public:
    struct Config {
        float entropy_weight = 0.4f;     ///< Attention entropy signal weight
        float cosine_weight = 0.3f;      ///< Hidden state cosine sim weight
        float variance_weight = 0.3f;    ///< Router logit variance weight
        float threshold = 0.6f;          ///< Shift confidence threshold (cautious)
        int cooldown_min = 2;            ///< Min tokens after switch before re-detecting
        int cooldown_max = 16;           ///< Max tokens after switch
    };

    struct ShiftResult {
        float confidence;          ///< 0.0 = no shift, 1.0 = definite shift
        bool detected;             ///< confidence > threshold (respects cooldown)
        float entropy_delta;       ///< Normalized change in attention entropy
        float cosine_sim;          ///< Cosine similarity of consecutive hidden states
        float variance_delta;      ///< Normalized change in router logit variance
    };

    ShiftDetector();
    explicit ShiftDetector(Config config);

    /// Evaluate shift based on three signals.
    ShiftResult evaluate(float attn_entropy, float cosine_sim,
                         float router_logit_variance);

    /// Compute adaptive cooldown: strong shift -> short cooldown, weak -> long.
    int cooldown_tokens(float confidence) const;

    /// Reset state (new sequence).
    void reset();

    const Config& config() const { return config_; }

private:
    Config config_;
    float prev_entropy_ = 0.0f;
    float prev_variance_ = 0.0f;
    int tokens_since_switch_ = 0;
    int current_cooldown_ = 0;
    bool first_token_ = true;
};

/// Compute cosine similarity between two vectors.
float cosine_similarity(const float* a, const float* b, int dim);

}  // namespace nos
