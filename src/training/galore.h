#pragma once

/// @file galore.h
/// @brief GaLore gradient low-rank projection optimizer.
///
/// Projects gradients of shared attention/router layers into a low-rank
/// subspace before the Adam update. This reduces optimizer memory by ~95%:
///   - P matrix: rows * rank * 4 bytes (e.g., 4096 * 128 * 4 = 2MB per weight)
///   - Compact Adam: 2 * rank * cols * 4 (e.g., 2 * 128 * 4096 * 4 = 4MB)
///   - vs Full Adam: 2 * rows * cols * 4 (e.g., 2 * 4096 * 4096 * 4 = 128MB)
///   - Savings: ~95% optimizer memory reduction

#include <cstddef>
#include <vector>

namespace nos {

struct AdamState;  // Forward declaration from badam.h

/// GaLore hyperparameters.
struct GaLoreConfig {
    int rank = 128;            ///< Rank for low-rank projection
    int update_interval = 200; ///< Recompute projection every T steps
    float scale_alpha = 0.25f; ///< Scaling factor for projected update
};

/// GaLore gradient low-rank projection optimizer.
///
/// Steps:
///   1. If step % update_interval == 0: recompute P via randomized truncated SVD
///   2. Project gradient to compact space: R = P^T * G (rank x cols)
///   3. Adam update on R using compact_state_ (m, v are rank x cols)
///   4. Project back: delta = P * adam_update(R)
///   5. W -= lr * scale_alpha * delta
class GaLoreOptimizer {
public:
    /// Construct for a weight matrix of given dimensions.
    GaLoreOptimizer(GaLoreConfig config, size_t rows, size_t cols);

    /// Perform one optimization step on a weight matrix.
    /// @param weight   Weight matrix (rows x cols), modified in-place
    /// @param gradient Gradient matrix (rows x cols)
    /// @param rows     Number of rows
    /// @param cols     Number of columns
    /// @param lr       Learning rate
    void step(float* weight, const float* gradient,
              size_t rows, size_t cols, float lr);

    /// Memory usage of projection matrix + compact Adam states.
    size_t memory_bytes() const;

    /// Current step counter.
    int step_count() const;

private:
    GaLoreConfig config_;
    size_t rows_;
    size_t cols_;
    size_t rank_;

    /// Projection matrix P (rows x rank)
    std::vector<float> P_;

    /// Compact Adam state (rank x cols sized)
    std::vector<float> compact_m_;
    std::vector<float> compact_v_;
    size_t adam_step_ = 0;

    int step_ = 0;

    /// Recompute projection matrix via randomized truncated SVD.
    void compute_projection(const float* gradient, size_t rows, size_t cols);

    /// Modified Gram-Schmidt QR decomposition.
    /// @param A   Input matrix (rows x cols)
    /// @param Q   Output Q matrix (rows x min(rows,cols)) -- orthonormal columns
    /// @param rows Number of rows
    /// @param cols Number of columns
    void qr_gram_schmidt(const float* A, float* Q,
                         size_t rows, size_t cols);

    /// Jacobi SVD for small matrices.
    /// @param A        Input matrix (rows x cols)
    /// @param U        Output left singular vectors (rows x min(rows,cols))
    /// @param S        Output singular values (min(rows,cols))
    /// @param rows     Number of rows
    /// @param cols     Number of columns
    /// @param max_iters Maximum Jacobi iterations
    void svd_jacobi(const float* A, float* U, float* S,
                    size_t rows, size_t cols, int max_iters = 100);
};

}  // namespace nos
