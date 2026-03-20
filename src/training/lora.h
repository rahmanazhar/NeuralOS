#pragma once

/// @file lora.h
/// @brief LoRA adapter with forward pass injection, backward pass, and merge.
///
/// LoRA (Low-Rank Adaptation) inserts trainable rank-r matrices A and B
/// alongside frozen base weights. The forward pass adds delta = (alpha/rank) * B * A * x.
/// Only A and B are trained, making fine-tuning memory-efficient.
///
/// After training, adapters can be merged into base weights permanently:
///   W_new = W_base + (alpha/rank) * B * A

#include <cstddef>
#include <string>
#include <vector>

namespace nos {

/// LoRA hyperparameters.
struct LoRAConfig {
    size_t rank = 16;
    float alpha = 16.0f;
    std::vector<std::string> target_layers;  ///< Default: all attention Q/K/V/O projections
};

/// LoRA adapter for a single weight matrix.
///
/// A is (rank x input_dim) with Kaiming He initialization.
/// B is (output_dim x rank) with zero initialization.
/// Forward: delta = (alpha/rank) * B * (A * x)
class LoRAAdapter {
public:
    /// Construct adapter for given dimensions.
    LoRAAdapter(LoRAConfig config, size_t input_dim, size_t output_dim);

    /// Default constructor for load().
    LoRAAdapter() = default;

    /// Forward pass: compute delta = (alpha/rank) * B * A * x.
    /// @param x       Input vector (input_dim)
    /// @param delta   Output delta vector (output_dim), added to base output
    /// @param batch_size  Number of vectors (default 1)
    void forward(const float* x, float* delta, size_t batch_size = 1) const;

    /// Backward pass through B*A.
    /// @param x           Input vector (input_dim) from forward pass
    /// @param grad_output Gradient w.r.t. output (output_dim)
    /// @param grad_A      Output: gradient for A (rank x input_dim)
    /// @param grad_B      Output: gradient for B (output_dim x rank)
    void backward(const float* x, const float* grad_output,
                  float* grad_A, float* grad_B) const;

    /// SGD update on A and B matrices.
    void update(const float* grad_A, const float* grad_B, float lr);

    /// Permanently merge adapter into base weight matrix.
    /// W += (alpha/rank) * B * A
    void merge_into(float* W, size_t rows, size_t cols) const;

    /// Save adapter to directory: lora_adapter.json + lora_A.bin + lora_B.bin.
    bool save(const std::string& dir) const;

    /// Load adapter from directory. Validates dimensions before overwriting.
    bool load(const std::string& dir);

    /// Rank of the adapter.
    size_t rank() const;

    /// Alpha scaling factor.
    float alpha() const;

    /// Total trainable parameter count: rank * (input_dim + output_dim).
    size_t param_count() const;

    /// Input dimension.
    size_t input_dim() const;

    /// Output dimension.
    size_t output_dim() const;

    /// Direct access to A matrix (for testing).
    float* A_data();
    const float* A_data() const;

    /// Direct access to B matrix (for testing).
    float* B_data();
    const float* B_data() const;

private:
    LoRAConfig config_;
    size_t input_dim_ = 0;
    size_t output_dim_ = 0;
    std::vector<float> A_;  ///< rank x input_dim, Kaiming He init
    std::vector<float> B_;  ///< output_dim x rank, zero init
};

}  // namespace nos
