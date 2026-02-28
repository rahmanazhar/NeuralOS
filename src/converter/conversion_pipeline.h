#pragma once

/// @file conversion_pipeline.h
/// @brief End-to-end model conversion pipeline orchestration.
///
/// Orchestrates: model reading -> activation collection -> clustering ->
/// quantization -> router calibration -> .nxp file writing with per-layer
/// checkpointing for resumability.

#include <cstdint>
#include <string>

namespace nos {

/// Configuration for the conversion pipeline.
struct ConversionConfig {
    std::string input_path;             ///< Path to model file or directory
    std::string output_dir;             ///< Output directory for .nxp and config
    int target_expert_size_mb = 100;    ///< Target expert size in MB
    int calibration_samples = 1024;     ///< Number of calibration samples
    int top_k = 2;                      ///< Experts selected per token
    bool resume = true;                 ///< Enable checkpoint-based resume
    std::string calibration_data_path;  ///< Path to calibration text file
                                        ///< Empty = synthetic mode (test only)
};

/// End-to-end model conversion pipeline.
class ConversionPipeline {
public:
    ConversionPipeline();
    ~ConversionPipeline();

    ConversionPipeline(const ConversionPipeline&) = delete;
    ConversionPipeline& operator=(const ConversionPipeline&) = delete;

    /// Run the full conversion pipeline.
    ///
    /// Stages:
    ///   1. Collect activations (calibration passes at FP16)
    ///   2. Per-layer: cluster, quantize, write experts
    ///   3. Router re-calibration
    ///   4. Write non-expert tensors (embeddings, output projection)
    ///   5. Finalize: write model_config.json
    ///
    /// @param config  Conversion configuration
    /// @return true on success
    bool run(const ConversionConfig& config);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace nos
