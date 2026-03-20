#pragma once

/// @file badam.h
/// @brief BAdam block-wise Adam optimizer for edge training.
///
/// BAdam trains one expert block at a time, keeping optimizer states
/// (first/second moments) only for the active block. After training a
/// block for steps_per_block steps, optimizer state is reset and the
/// next block is loaded. This keeps peak memory to a single block's
/// weights + Adam states + gradients + activations.
///
/// Block-wise training pattern (used by Trainer):
///   1. For each block (layer, expert_id): load weights from NVMe to FP32 RAM
///   2. init_state(block.num_params)
///   3. For steps_per_block steps: compute gradient, call step()
///   4. Quantize updated FP32 weights back to ternary, save to NVMe
///   5. reset_state() -- free optimizer memory before next block

#include <cstddef>
#include <vector>

namespace nos {

/// First/second moment state for Adam optimizer with step counter.
struct AdamState {
    std::vector<float> m;   ///< First moment (mean of gradients)
    std::vector<float> v;   ///< Second moment (mean of squared gradients)
    size_t step = 0;        ///< Step counter for bias correction
};

/// BAdam hyperparameters.
struct BAdamConfig {
    float lr = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    int steps_per_block = 100;
};

/// BAdam block-wise Adam optimizer.
///
/// Implements AdamW (decoupled weight decay) with per-block state lifecycle.
/// Between blocks, reset_state() zeroes out moments and resets the step
/// counter, freeing the optimizer memory for the next block.
class BAdamOptimizer {
public:
    explicit BAdamOptimizer(BAdamConfig config);

    /// Allocate m and v vectors initialized to 0 for num_params parameters.
    void init_state(size_t num_params);

    /// AdamW update step.
    ///   m = beta1*m + (1-beta1)*g
    ///   v = beta2*v + (1-beta2)*g*g
    ///   m_hat = m / (1 - beta1^t)
    ///   v_hat = v / (1 - beta2^t)
    ///   w = w * (1 - lr*weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
    void step(float* weights, const float* gradients, size_t num_params);

    /// Zero out m, v, and reset step counter (called between blocks).
    void reset_state();

    /// Returns 2 * num_params * sizeof(float) for budget reporting.
    size_t memory_bytes() const;

    /// Current step counter.
    size_t step_count() const;

    /// Access config.
    const BAdamConfig& config() const;

private:
    BAdamConfig config_;
    AdamState state_;
};

}  // namespace nos
