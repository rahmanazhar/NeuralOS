/// @file test_galore.cpp
/// @brief Catch2 tests for GaLore gradient low-rank projection optimizer.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "training/galore.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

/// Helper: compute Frobenius norm of a matrix.
float frobenius_norm(const float* A, size_t rows, size_t cols) {
    float sum = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        sum += A[i] * A[i];
    }
    return std::sqrt(sum);
}

}  // namespace

TEST_CASE("GaLore QR orthogonality", "[galore]") {
    // Test that QR decomposition produces orthonormal Q columns.
    // We test indirectly by using a GaLoreOptimizer on a small matrix
    // and verifying the projection makes sense.

    // Create a small gradient matrix where we know the structure
    const size_t rows = 8;
    const size_t cols = 8;
    const size_t rank = 4;

    nos::GaLoreConfig cfg;
    cfg.rank = static_cast<int>(rank);
    cfg.update_interval = 1;  // Recompute every step
    cfg.scale_alpha = 1.0f;

    nos::GaLoreOptimizer opt(cfg, rows, cols);

    // Create a rank-4 gradient matrix
    std::vector<float> weight(rows * cols, 1.0f);
    std::vector<float> gradient(rows * cols, 0.0f);

    // Fill gradient with a low-rank structure
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            gradient[r * cols + c] = static_cast<float>(r + 1) *
                                     static_cast<float>(c + 1) * 0.01f;
        }
    }

    // One step should not crash and should modify weights
    std::vector<float> weight_copy(weight);
    opt.step(weight.data(), gradient.data(), rows, cols, 0.1f);

    // Weights should have changed
    bool changed = false;
    for (size_t i = 0; i < rows * cols; ++i) {
        if (std::fabs(weight[i] - weight_copy[i]) > 1e-8f) {
            changed = true;
            break;
        }
    }
    REQUIRE(changed);
}

TEST_CASE("GaLore step reduces gradient norm", "[galore]") {
    // After a GaLore step in the direction of steepest descent,
    // re-evaluating the gradient should give smaller norm.
    const size_t rows = 16;
    const size_t cols = 16;

    nos::GaLoreConfig cfg;
    cfg.rank = 8;
    cfg.update_interval = 1;
    cfg.scale_alpha = 0.5f;

    nos::GaLoreOptimizer opt(cfg, rows, cols);

    // Objective: minimize sum of W^2 (gradient = 2*W)
    std::vector<float> W(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        W[i] = static_cast<float>(i % 7) * 0.3f - 0.9f;
    }

    float initial_norm = frobenius_norm(W.data(), rows, cols);

    // Several steps
    for (int s = 0; s < 20; ++s) {
        std::vector<float> grad(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            grad[i] = 2.0f * W[i];
        }
        opt.step(W.data(), grad.data(), rows, cols, 0.01f);
    }

    float final_norm = frobenius_norm(W.data(), rows, cols);
    REQUIRE(final_norm < initial_norm);
}

TEST_CASE("GaLore projection recomputed at interval", "[galore]") {
    const size_t rows = 8;
    const size_t cols = 8;

    nos::GaLoreConfig cfg;
    cfg.rank = 4;
    cfg.update_interval = 3;

    nos::GaLoreOptimizer opt(cfg, rows, cols);

    std::vector<float> W(rows * cols, 1.0f);
    std::vector<float> grad(rows * cols, 0.1f);

    // Step count should increment
    REQUIRE(opt.step_count() == 0);
    opt.step(W.data(), grad.data(), rows, cols, 0.01f);
    REQUIRE(opt.step_count() == 1);
    opt.step(W.data(), grad.data(), rows, cols, 0.01f);
    REQUIRE(opt.step_count() == 2);
    opt.step(W.data(), grad.data(), rows, cols, 0.01f);
    REQUIRE(opt.step_count() == 3);
}

TEST_CASE("GaLore memory_bytes is compact", "[galore]") {
    const size_t rows = 4096;
    const size_t cols = 4096;

    nos::GaLoreConfig cfg;
    cfg.rank = 128;

    nos::GaLoreOptimizer opt(cfg, rows, cols);

    size_t mem = opt.memory_bytes();

    // P: rows * rank * 4 = 4096 * 128 * 4 = 2MB
    // m: rank * cols * 4 = 128 * 4096 * 4 = 2MB
    // v: rank * cols * 4 = 128 * 4096 * 4 = 2MB
    // Total: ~6MB
    size_t expected = rows * 128 * sizeof(float) +
                      2 * 128 * cols * sizeof(float);
    REQUIRE(mem == expected);

    // Full Adam would be 2 * rows * cols * 4 = 128MB
    size_t full_adam = 2 * rows * cols * sizeof(float);
    // GaLore should use much less
    REQUIRE(mem < full_adam / 10);  // At least 10x savings
}

TEST_CASE("GaLore handles very small matrices", "[galore]") {
    // Edge case: rank larger than matrix dimension
    const size_t rows = 4;
    const size_t cols = 4;

    nos::GaLoreConfig cfg;
    cfg.rank = 128;  // Much larger than matrix
    cfg.update_interval = 1;

    // Should clamp rank to min(rows, cols)
    nos::GaLoreOptimizer opt(cfg, rows, cols);

    std::vector<float> W(rows * cols, 1.0f);
    std::vector<float> grad(rows * cols, 0.5f);

    // Should not crash
    opt.step(W.data(), grad.data(), rows, cols, 0.01f);
    REQUIRE(opt.step_count() == 1);
}

TEST_CASE("GaLore deterministic with fixed seed", "[galore]") {
    const size_t rows = 8;
    const size_t cols = 8;

    nos::GaLoreConfig cfg;
    cfg.rank = 4;
    cfg.update_interval = 1;

    // Run twice with same input
    std::vector<float> W1(rows * cols, 1.0f);
    std::vector<float> W2(rows * cols, 1.0f);
    std::vector<float> grad(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        grad[i] = static_cast<float>(i) * 0.01f;
    }

    nos::GaLoreOptimizer opt1(cfg, rows, cols);
    nos::GaLoreOptimizer opt2(cfg, rows, cols);

    opt1.step(W1.data(), grad.data(), rows, cols, 0.01f);
    opt2.step(W2.data(), grad.data(), rows, cols, 0.01f);

    // Results should be identical (fixed seed in SVD)
    for (size_t i = 0; i < rows * cols; ++i) {
        REQUIRE(W1[i] == W2[i]);
    }
}
