/// @file galore.cpp
/// @brief GaLore gradient low-rank projection optimizer implementation.
///
/// Hand-rolled randomized SVD and QR decomposition per research requirement
/// ("training algorithms MUST be hand-rolled" -- no Eigen dependency).

#include "training/galore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace nos {

GaLoreOptimizer::GaLoreOptimizer(GaLoreConfig config, size_t rows, size_t cols)
    : config_(config)
    , rows_(rows)
    , cols_(cols)
    , rank_(static_cast<size_t>(std::min(config.rank, static_cast<int>(std::min(rows, cols)))))
{
    // Projection matrix: rows x rank
    P_.resize(rows_ * rank_, 0.0f);

    // Compact Adam states: rank x cols
    const size_t compact_size = rank_ * cols_;
    compact_m_.assign(compact_size, 0.0f);
    compact_v_.assign(compact_size, 0.0f);
    adam_step_ = 0;
}

void GaLoreOptimizer::step(float* weight, const float* gradient,
                           size_t rows, size_t cols, float lr) {
    // Step 1: Recompute projection if needed
    if (step_ % config_.update_interval == 0) {
        compute_projection(gradient, rows, cols);
    }

    // Step 2: Project gradient to compact space: R = P^T * G (rank x cols)
    // P is rows x rank, G is rows x cols => P^T * G is rank x cols
    std::vector<float> R(rank_ * cols, 0.0f);
    for (size_t r = 0; r < rank_; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < rows; ++k) {
                sum += P_[k * rank_ + r] * gradient[k * cols + c];
            }
            R[r * cols + c] = sum;
        }
    }

    // Step 3: Adam update on R in compact space
    ++adam_step_;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float t = static_cast<float>(adam_step_);
    const float bc1 = 1.0f - std::pow(beta1, t);
    const float bc2 = 1.0f - std::pow(beta2, t);

    std::vector<float> adam_update(rank_ * cols);
    for (size_t i = 0; i < rank_ * cols; ++i) {
        const float g = R[i];
        compact_m_[i] = beta1 * compact_m_[i] + (1.0f - beta1) * g;
        compact_v_[i] = beta2 * compact_v_[i] + (1.0f - beta2) * g * g;

        const float m_hat = compact_m_[i] / bc1;
        const float v_hat = compact_v_[i] / bc2;
        adam_update[i] = m_hat / (std::sqrt(v_hat) + eps);
    }

    // Step 4: Project back: delta = P * adam_update (rows x cols)
    // P is rows x rank, adam_update is rank x cols => delta is rows x cols
    // Step 5: W -= lr * scale_alpha * delta
    const float scale = lr * config_.scale_alpha;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float delta = 0.0f;
            for (size_t k = 0; k < rank_; ++k) {
                delta += P_[r * rank_ + k] * adam_update[k * cols + c];
            }
            weight[r * cols + c] -= scale * delta;
        }
    }

    ++step_;
}

size_t GaLoreOptimizer::memory_bytes() const {
    // P matrix + compact m + compact v
    const size_t p_bytes = rows_ * rank_ * sizeof(float);
    const size_t compact_bytes = 2 * rank_ * cols_ * sizeof(float);
    return p_bytes + compact_bytes;
}

int GaLoreOptimizer::step_count() const {
    return step_;
}

void GaLoreOptimizer::compute_projection(const float* gradient,
                                         size_t rows, size_t cols) {
    // Randomized truncated SVD of gradient matrix.
    // Fixed seed for reproducibility.
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    const size_t oversample = 10;
    const size_t sample_cols = rank_ + oversample;
    const size_t actual_sample = std::min(sample_cols, cols);

    // 1. Generate random Gaussian matrix Omega (cols x actual_sample)
    std::vector<float> Omega(cols * actual_sample);
    for (size_t i = 0; i < cols * actual_sample; ++i) {
        Omega[i] = normal(rng);
    }

    // 2. Y = G * Omega (rows x actual_sample)
    std::vector<float> Y(rows * actual_sample, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < actual_sample; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols; ++k) {
                sum += gradient[r * cols + k] * Omega[k * actual_sample + c];
            }
            Y[r * actual_sample + c] = sum;
        }
    }

    // 3. QR decomposition of Y: Q (rows x rank)
    const size_t qr_cols = std::min(actual_sample, rows);
    std::vector<float> Q(rows * qr_cols, 0.0f);
    qr_gram_schmidt(Y.data(), Q.data(), rows, actual_sample);

    // Use only rank columns of Q
    const size_t use_cols = std::min(rank_, qr_cols);

    // 4. B = Q^T * G (use_cols x cols)
    std::vector<float> B(use_cols * cols, 0.0f);
    for (size_t r = 0; r < use_cols; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < rows; ++k) {
                sum += Q[k * qr_cols + r] * gradient[k * cols + c];
            }
            B[r * cols + c] = sum;
        }
    }

    // 5. SVD of B via Jacobi iterations
    const size_t svd_min = std::min(use_cols, cols);
    std::vector<float> U_B(use_cols * svd_min, 0.0f);
    std::vector<float> S_B(svd_min, 0.0f);
    svd_jacobi(B.data(), U_B.data(), S_B.data(), use_cols, cols);

    // 6. P = Q * U_B[:, :rank] (rows x rank)
    // Q is rows x qr_cols, U_B is use_cols x svd_min
    // We take first rank_ columns of U_B
    const size_t take_cols = std::min(rank_, svd_min);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < rank_; ++c) {
            if (c < take_cols) {
                float sum = 0.0f;
                for (size_t k = 0; k < use_cols; ++k) {
                    sum += Q[r * qr_cols + k] * U_B[k * svd_min + c];
                }
                P_[r * rank_ + c] = sum;
            } else {
                P_[r * rank_ + c] = 0.0f;
            }
        }
    }
}

void GaLoreOptimizer::qr_gram_schmidt(const float* A, float* Q,
                                       size_t rows, size_t cols) {
    // Modified Gram-Schmidt for numerical stability.
    // Produces Q with orthonormal columns.
    const size_t out_cols = std::min(rows, cols);

    // Copy A into Q (only out_cols columns)
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < out_cols; ++c) {
            if (c < cols) {
                Q[r * out_cols + c] = A[r * cols + c];
            } else {
                Q[r * out_cols + c] = 0.0f;
            }
        }
    }

    for (size_t j = 0; j < out_cols; ++j) {
        // Normalize column j
        float norm = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            const float val = Q[r * out_cols + j];
            norm += val * val;
        }
        norm = std::sqrt(norm);

        if (norm < 1e-10f) {
            // Near-zero column: set to zero
            for (size_t r = 0; r < rows; ++r) {
                Q[r * out_cols + j] = 0.0f;
            }
            continue;
        }

        const float inv_norm = 1.0f / norm;
        for (size_t r = 0; r < rows; ++r) {
            Q[r * out_cols + j] *= inv_norm;
        }

        // Orthogonalize subsequent columns against column j
        for (size_t k = j + 1; k < out_cols; ++k) {
            float dot = 0.0f;
            for (size_t r = 0; r < rows; ++r) {
                dot += Q[r * out_cols + j] * Q[r * out_cols + k];
            }
            for (size_t r = 0; r < rows; ++r) {
                Q[r * out_cols + k] -= dot * Q[r * out_cols + j];
            }
        }
    }
}

void GaLoreOptimizer::svd_jacobi(const float* A, float* U, float* S,
                                  size_t rows, size_t cols,
                                  int max_iters) {
    // Jacobi one-sided SVD for small matrices.
    // Computes A = U * diag(S) * V^T, but we only need U and S.
    //
    // Strategy: compute A^T * A eigendecomposition via Jacobi rotations,
    // then recover U = A * V * diag(1/S).
    //
    // For small matrices (rank x cols where rank ~ 128), this is efficient.

    const size_t min_dim = std::min(rows, cols);

    // Work on a copy (rows x cols)
    std::vector<float> W(rows * cols);
    std::memcpy(W.data(), A, rows * cols * sizeof(float));

    // Initialize U as identity-like (rows x min_dim)
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < min_dim; ++c) {
            U[r * min_dim + c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    // One-sided Jacobi: apply rotations to columns of W to make W^T*W diagonal
    for (int iter = 0; iter < max_iters; ++iter) {
        float off_norm = 0.0f;

        for (size_t p = 0; p < min_dim; ++p) {
            for (size_t q = p + 1; q < cols; ++q) {
                // Compute 2x2 sub-problem: columns p and q of W
                float app = 0.0f, aqq = 0.0f, apq = 0.0f;
                for (size_t r = 0; r < rows; ++r) {
                    const float wp = W[r * cols + p];
                    const float wq = W[r * cols + q];
                    app += wp * wp;
                    aqq += wq * wq;
                    apq += wp * wq;
                }

                off_norm += apq * apq;

                if (std::fabs(apq) < 1e-12f) continue;

                // Compute Jacobi rotation angle
                const float tau = (aqq - app) / (2.0f * apq);
                float t;
                if (tau >= 0.0f) {
                    t = 1.0f / (tau + std::sqrt(1.0f + tau * tau));
                } else {
                    t = -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
                }
                const float cs = 1.0f / std::sqrt(1.0f + t * t);
                const float sn = t * cs;

                // Apply rotation to columns p and q of W
                for (size_t r = 0; r < rows; ++r) {
                    const float wp = W[r * cols + p];
                    const float wq = W[r * cols + q];
                    W[r * cols + p] = cs * wp - sn * wq;
                    W[r * cols + q] = sn * wp + cs * wq;
                }

                // Apply rotation to columns p and q of U (if within range)
                if (q < min_dim) {
                    for (size_t r = 0; r < rows; ++r) {
                        const float up = U[r * min_dim + p];
                        const float uq = U[r * min_dim + q];
                        U[r * min_dim + p] = cs * up - sn * uq;
                        U[r * min_dim + q] = sn * up + cs * uq;
                    }
                }
            }
        }

        // Convergence check
        if (off_norm < 1e-20f) break;
    }

    // Extract singular values from column norms of W
    // and normalize U columns
    for (size_t c = 0; c < min_dim; ++c) {
        float col_norm = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            const float val = W[r * cols + c];
            col_norm += val * val;
        }
        col_norm = std::sqrt(col_norm);
        S[c] = col_norm;

        // U column = W column / singular value
        if (col_norm > 1e-10f) {
            const float inv = 1.0f / col_norm;
            for (size_t r = 0; r < rows; ++r) {
                U[r * min_dim + c] = W[r * cols + c] * inv;
            }
        }
    }
}

}  // namespace nos
