#pragma once

/// @file activation_collector.h
/// @brief Activation magnitude collection for expert clustering.
///
/// Collects per-neuron activation magnitudes by running calibration data
/// through FFN weights at FP16 precision. Uses real calibration text when
/// available, or synthetic embedding lookups for testing.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nos {

class ModelReader;

/// Forward declaration for ConversionConfig (defined in conversion_pipeline.h).
/// Using a minimal struct here to avoid circular dependency.
struct ActivationCollectorConfig {
    std::string calibration_data_path;
    int calibration_samples = 1024;
};

/// Per-layer activation magnitude data.
struct ActivationData {
    /// Per-layer L2 norms of each neuron's activation across calibration samples.
    /// Indexed as [layer][neuron_index].
    std::vector<std::vector<float>> per_layer_magnitudes;
};

/// Collect activation magnitudes for expert clustering.
///
/// @param reader  Model reader with loaded weights
/// @param config  Activation collection configuration
/// @return Per-layer neuron activation magnitudes
ActivationData collect_activations(ModelReader& reader,
                                    const ActivationCollectorConfig& config);

/// Dense FP16 matrix-vector multiply for calibration passes.
///
/// Reads FP16 weights, converts to FP32, accumulates in FP32.
/// NOT the ternary kernel -- we need FP16 precision for clustering.
///
/// @param W     FP16 weight matrix [rows x cols], row-major
/// @param x     FP32 input vector [cols]
/// @param out   FP32 output vector [rows]
/// @param rows  Number of output rows
/// @param cols  Number of input columns
void dense_matvec_fp16(const uint16_t* W, const float* x, float* out,
                        int rows, int cols);

}  // namespace nos
