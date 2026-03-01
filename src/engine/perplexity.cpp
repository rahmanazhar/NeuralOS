/// @file perplexity.cpp
/// @brief Perplexity computation using non-overlapping context chunks.

#include "engine/perplexity.h"
#include "engine/inference_engine.h"

#include <cmath>
#include <cstdio>

namespace nos {

double compute_perplexity(InferenceEngine& engine,
                          const std::vector<int>& tokens,
                          int context_length) {
    if (tokens.size() < 2) return -1.0;

    int ctx_len = context_length > 0
                  ? context_length
                  : static_cast<int>(engine.config().max_seq_len);

    int vocab_size = engine.vocab_size();
    int n_tokens = static_cast<int>(tokens.size());

    double total_loss = 0.0;
    int total_predictions = 0;

    // Process in non-overlapping chunks
    for (int chunk_start = 0; chunk_start < n_tokens - 1; chunk_start += ctx_len) {
        engine.reset_kv_cache();

        int chunk_end = std::min(chunk_start + ctx_len, n_tokens - 1);
        int chunk_len = chunk_end - chunk_start;

        for (int i = 0; i < chunk_len; i++) {
            int token = tokens[static_cast<size_t>(chunk_start + i)];
            int next_token = tokens[static_cast<size_t>(chunk_start + i + 1)];

            const float* logits = engine.forward_step(token, i);
            if (!logits) continue;

            // Compute log-softmax for the next token
            // Find max for numerical stability
            float max_logit = logits[0];
            for (int v = 1; v < vocab_size; v++) {
                if (logits[v] > max_logit) max_logit = logits[v];
            }

            double sum_exp = 0.0;
            for (int v = 0; v < vocab_size; v++) {
                sum_exp += std::exp(static_cast<double>(logits[v]) - static_cast<double>(max_logit));
            }

            double log_prob = static_cast<double>(logits[next_token])
                            - static_cast<double>(max_logit)
                            - std::log(sum_exp);

            total_loss -= log_prob;
            total_predictions++;
        }
    }

    if (total_predictions == 0) return -1.0;

    double mean_loss = total_loss / static_cast<double>(total_predictions);
    return std::exp(mean_loss);
}

}  // namespace nos
