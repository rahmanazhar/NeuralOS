#pragma once

/// @file router_calibrator.h
/// @brief Router weight initialization and post-quantization re-calibration.
///
/// Initializes router gating weights from clustering centroids, then
/// re-calibrates on quantized experts to minimize reconstruction error.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "converter/kmeans.h"
#include "converter/quantizer.h"

namespace nos {

/// Router weights for one layer.
struct RouterWeights {
    std::vector<float> weights;  ///< [num_experts x hidden_dim], row-major
    int num_experts = 0;
    int hidden_dim = 0;
};

/// Initialize router weights from clustering centroids.
///
/// Projects cluster centroids to create initial router weight matrix.
/// The centroid of each cluster in activation space provides a natural
/// routing signal for which tokens should be sent to which expert.
///
/// @param clusters       K-means clustering result
/// @param activation_data  Neuron activation magnitudes [n_neurons]
/// @param n_neurons       Number of neurons clustered
/// @param hidden_dim     Model hidden dimension
/// @return Initial router weights [num_experts x hidden_dim]
RouterWeights init_from_centroids(const KMeansResult& clusters,
                                   const float* activation_data, int n_neurons,
                                   int hidden_dim);

/// Re-calibrate router weights on quantized experts.
///
/// Runs calibration passes through quantized expert weights and adjusts
/// router weights to minimize reconstruction error vs. the original dense
/// FFN output. Uses gradient-free optimization (proportional weight update).
///
/// @param router         Router weights to update (modified in-place)
/// @param quantized_experts  Quantized expert weights for this layer
/// @param calibration_inputs  Calibration input vectors [n_samples x hidden_dim]
/// @param n_samples      Number of calibration samples
void recalibrate_router(RouterWeights& router,
                         const std::vector<QuantizedWeights>& quantized_experts,
                         const float* calibration_inputs, int n_samples);

}  // namespace nos
