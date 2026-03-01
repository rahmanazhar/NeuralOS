#pragma once

/// @file sampling.h
/// @brief Token sampling suite: repetition penalty, temperature, top-k, top-p, min-p.

#include <cstdint>
#include <random>
#include <vector>

namespace nos {

struct SamplingParams {
    float temperature = 1.0f;         // 0 = greedy
    int top_k = 40;                   // 0 = disabled
    float top_p = 0.95f;             // 1.0 = disabled
    float repetition_penalty = 1.1f;  // 1.0 = disabled
    float min_p = 0.05f;             // 0.0 = disabled
    uint64_t seed = 0;               // 0 = random seed
};

/// Apply repetition penalty to logits for tokens in context.
void apply_repetition_penalty(float* logits, int vocab_size,
                              const std::vector<int>& context_ids, float penalty);

/// Scale logits by 1/temperature.
void apply_temperature(float* logits, int vocab_size, float temp);

/// Zero out all logits below the top-k threshold.
void apply_top_k(float* logits, int vocab_size, int k);

/// Zero out logits that fall below cumulative probability p.
void apply_top_p(float* logits, int vocab_size, float p);

/// Zero out logits below min_p fraction of the maximum probability.
void apply_min_p(float* logits, int vocab_size, float min_p_val);

/// Greedy decode: return argmax token.
int greedy_sample(const float* logits, int vocab_size);

/// Categorical sample from logits (softmax internally).
int sample_token(const float* logits, int vocab_size, std::mt19937& rng);

/// Full sampling pipeline: applies all transforms, then samples.
int sample(float* logits, int vocab_size, const SamplingParams& params,
           const std::vector<int>& context, std::mt19937& rng);

}  // namespace nos
