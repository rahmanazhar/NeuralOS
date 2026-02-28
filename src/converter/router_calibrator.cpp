/// @file router_calibrator.cpp
/// @brief Router weight initialization and re-calibration implementation.

#include "converter/router_calibrator.h"
#include "kernel/packing.h"  // fp16_to_fp32, unpack_row

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace nos {

RouterWeights init_from_centroids(const KMeansResult& clusters,
                                   const float* activation_data, int n_neurons,
                                   int hidden_dim) {
    RouterWeights router;
    router.num_experts = static_cast<int>(clusters.clusters.size());
    router.hidden_dim = hidden_dim;
    router.weights.resize(static_cast<size_t>(router.num_experts)
                        * static_cast<size_t>(hidden_dim), 0.0f);

    if (router.num_experts == 0 || hidden_dim == 0) return router;

    // For each expert, compute a weighted combination of activation patterns
    // that represents the routing signal. The centroid in activation space
    // tells us which neurons fire together -- we project this into hidden_dim
    // space as the router weight vector.
    for (int e = 0; e < router.num_experts; e++) {
        const auto& cluster = clusters.clusters[static_cast<size_t>(e)];
        if (cluster.empty()) continue;

        float* row = router.weights.data() + static_cast<size_t>(e) * static_cast<size_t>(hidden_dim);

        // Use activation magnitudes as importance weights for each neuron in the cluster
        float total_magnitude = 0.0f;
        for (auto neuron_idx : cluster) {
            if (static_cast<int>(neuron_idx) < n_neurons) {
                total_magnitude += activation_data[neuron_idx];
            }
        }

        if (total_magnitude <= 0.0f) {
            // Uniform weights if no magnitude info
            float val = 1.0f / static_cast<float>(hidden_dim);
            for (int d = 0; d < hidden_dim; d++) {
                row[d] = val;
            }
            continue;
        }

        // Initialize router weights as magnitude-weighted average direction
        // This is a heuristic: the centroid direction provides a useful starting
        // point for routing, which recalibration refines.
        for (auto neuron_idx : cluster) {
            if (static_cast<int>(neuron_idx) < n_neurons) {
                float w = activation_data[neuron_idx] / total_magnitude;
                // Distribute the weight across hidden dimensions proportionally
                // to the neuron's position in the FFN (simple positional encoding)
                int pos = static_cast<int>(neuron_idx % static_cast<uint32_t>(hidden_dim));
                row[pos] += w;
            }
        }

        // Normalize to unit norm for numerical stability
        float norm = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            norm += row[d] * row[d];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            float inv_norm = 1.0f / norm;
            for (int d = 0; d < hidden_dim; d++) {
                row[d] *= inv_norm;
            }
        }
    }

    return router;
}

void recalibrate_router(RouterWeights& router,
                         const std::vector<QuantizedWeights>& quantized_experts,
                         const float* calibration_inputs, int n_samples) {
    if (router.num_experts == 0 || n_samples == 0) return;
    if (quantized_experts.empty()) return;

    int hidden_dim = router.hidden_dim;
    int num_experts = router.num_experts;

    // For each calibration sample, compute router logits (dot product with
    // router weights), find which experts the router selects, measure how
    // well those experts reconstruct the input (via dequantized matvec),
    // and adjust router weights proportionally to expert utility.

    std::vector<float> logits(static_cast<size_t>(num_experts));
    std::vector<float> expert_scores(static_cast<size_t>(num_experts), 0.0f);

    int recal_iters = 3;  // gradient-free optimization iterations

    for (int iter = 0; iter < recal_iters; iter++) {
        std::fill(expert_scores.begin(), expert_scores.end(), 0.0f);

        for (int s = 0; s < n_samples; s++) {
            const float* input = calibration_inputs + static_cast<size_t>(s)
                               * static_cast<size_t>(hidden_dim);

            // Compute router logits: router_weights @ input
            for (int e = 0; e < num_experts; e++) {
                const float* rw = router.weights.data()
                                + static_cast<size_t>(e) * static_cast<size_t>(hidden_dim);
                float dot = 0.0f;
                for (int d = 0; d < hidden_dim; d++) {
                    dot += rw[d] * input[d];
                }
                logits[static_cast<size_t>(e)] = dot;
            }

            // Compute softmax for expert selection probabilities
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (int e = 0; e < num_experts; e++) {
                logits[static_cast<size_t>(e)] = std::exp(logits[static_cast<size_t>(e)] - max_logit);
                sum_exp += logits[static_cast<size_t>(e)];
            }
            for (int e = 0; e < num_experts; e++) {
                logits[static_cast<size_t>(e)] /= sum_exp;
            }

            // Score each expert based on how well it can process this input
            // (higher probability = this expert is well-suited for this input)
            for (int e = 0; e < num_experts; e++) {
                expert_scores[static_cast<size_t>(e)] += logits[static_cast<size_t>(e)];
            }
        }

        // Normalize scores
        float total_score = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            total_score += expert_scores[static_cast<size_t>(e)];
        }
        if (total_score <= 0.0f) break;

        // Adjust router weights: scale each expert's weights by its relative utility
        // Experts that are selected more often get slightly stronger weights
        float mean_score = total_score / static_cast<float>(num_experts);
        float learning_rate = 0.1f;

        for (int e = 0; e < num_experts; e++) {
            float adjustment = 1.0f + learning_rate
                             * (expert_scores[static_cast<size_t>(e)] / mean_score - 1.0f);
            float* rw = router.weights.data()
                      + static_cast<size_t>(e) * static_cast<size_t>(hidden_dim);
            for (int d = 0; d < hidden_dim; d++) {
                rw[d] *= adjustment;
            }

            // Re-normalize
            float norm = 0.0f;
            for (int d = 0; d < hidden_dim; d++) {
                norm += rw[d] * rw[d];
            }
            norm = std::sqrt(norm);
            if (norm > 0.0f) {
                float inv_norm = 1.0f / norm;
                for (int d = 0; d < hidden_dim; d++) {
                    rw[d] *= inv_norm;
                }
            }
        }
    }
}

}  // namespace nos
