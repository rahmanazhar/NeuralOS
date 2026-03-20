/// @file test_lora.cpp
/// @brief Catch2 tests for LoRA adapter.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "training/lora.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <vector>

TEST_CASE("LoRA B zero init: initial forward returns zero delta", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 4;
    cfg.alpha = 4.0f;

    const size_t in_dim = 8;
    const size_t out_dim = 8;
    nos::LoRAAdapter adapter(cfg, in_dim, out_dim);

    std::vector<float> x(in_dim, 1.0f);
    std::vector<float> delta(out_dim, 999.0f);

    adapter.forward(x.data(), delta.data());

    // B is zero-initialized, so delta should be all zeros
    for (size_t i = 0; i < out_dim; ++i) {
        REQUIRE(delta[i] == 0.0f);
    }
}

TEST_CASE("LoRA forward with manual A and B values", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 2;
    cfg.alpha = 2.0f;

    const size_t in_dim = 3;
    const size_t out_dim = 3;
    nos::LoRAAdapter adapter(cfg, in_dim, out_dim);

    // Set A manually: rank x input_dim = 2 x 3
    // A = [[1, 0, 0],
    //      [0, 1, 0]]
    float* A = adapter.A_data();
    std::memset(A, 0, 2 * 3 * sizeof(float));
    A[0 * 3 + 0] = 1.0f;  // A[0][0] = 1
    A[1 * 3 + 1] = 1.0f;  // A[1][1] = 1

    // Set B manually: output_dim x rank = 3 x 2
    // B = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    float* B = adapter.B_data();
    std::memset(B, 0, 3 * 2 * sizeof(float));
    B[0 * 2 + 0] = 1.0f;  // B[0][0] = 1
    B[1 * 2 + 1] = 1.0f;  // B[1][1] = 1
    B[2 * 2 + 0] = 1.0f;  // B[2][0] = 1
    B[2 * 2 + 1] = 1.0f;  // B[2][1] = 1

    // x = [2, 3, 5]
    std::vector<float> x = {2.0f, 3.0f, 5.0f};
    std::vector<float> delta(out_dim, 0.0f);

    adapter.forward(x.data(), delta.data());

    // A * x = [2, 3]
    // B * (A*x) = [1*2+0*3, 0*2+1*3, 1*2+1*3] = [2, 3, 5]
    // scale = alpha / rank = 2/2 = 1
    // delta = 1 * [2, 3, 5] = [2, 3, 5]
    REQUIRE_THAT(delta[0], Catch::Matchers::WithinAbs(2.0, 1e-5));
    REQUIRE_THAT(delta[1], Catch::Matchers::WithinAbs(3.0, 1e-5));
    REQUIRE_THAT(delta[2], Catch::Matchers::WithinAbs(5.0, 1e-5));
}

TEST_CASE("LoRA merge_into matches manual computation", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 2;
    cfg.alpha = 4.0f;

    const size_t in_dim = 3;
    const size_t out_dim = 3;
    nos::LoRAAdapter adapter(cfg, in_dim, out_dim);

    // Set simple A and B
    float* A = adapter.A_data();
    float* B = adapter.B_data();
    // A = identity-like 2x3: [[1,0,0],[0,1,0]]
    std::memset(A, 0, 2 * 3 * sizeof(float));
    A[0] = 1.0f;  // A[0][0]
    A[4] = 1.0f;  // A[1][1]
    // B = 3x2: [[1,0],[0,1],[0,0]]
    std::memset(B, 0, 3 * 2 * sizeof(float));
    B[0] = 1.0f;  // B[0][0]
    B[3] = 1.0f;  // B[1][1]

    // W starts as zeros
    std::vector<float> W(out_dim * in_dim, 0.0f);

    adapter.merge_into(W.data(), out_dim, in_dim);

    // B * A = [[1,0,0],[0,1,0],[0,0,0]]
    // scale = alpha/rank = 4/2 = 2
    // W += 2 * [[1,0,0],[0,1,0],[0,0,0]]
    REQUIRE_THAT(W[0], Catch::Matchers::WithinAbs(2.0, 1e-5));
    REQUIRE_THAT(W[4], Catch::Matchers::WithinAbs(2.0, 1e-5));
    REQUIRE_THAT(W[1], Catch::Matchers::WithinAbs(0.0, 1e-5));
    REQUIRE_THAT(W[8], Catch::Matchers::WithinAbs(0.0, 1e-5));
}

TEST_CASE("LoRA save/load round-trip", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 4;
    cfg.alpha = 8.0f;
    cfg.target_layers = {"q_proj", "v_proj"};

    const size_t in_dim = 16;
    const size_t out_dim = 16;
    nos::LoRAAdapter orig(cfg, in_dim, out_dim);

    // Set B to non-zero for more interesting round-trip
    float* B = orig.B_data();
    for (size_t i = 0; i < out_dim * cfg.rank; ++i) {
        B[i] = static_cast<float>(i) * 0.01f;
    }

    auto dir = std::filesystem::temp_directory_path() / "neuralos_test" / "lora_roundtrip";
    std::filesystem::remove_all(dir);

    REQUIRE(orig.save(dir.string()));

    nos::LoRAAdapter loaded;
    REQUIRE(loaded.load(dir.string()));

    REQUIRE(loaded.rank() == orig.rank());
    REQUIRE(loaded.alpha() == orig.alpha());
    REQUIRE(loaded.input_dim() == orig.input_dim());
    REQUIRE(loaded.output_dim() == orig.output_dim());
    REQUIRE(loaded.param_count() == orig.param_count());

    // Compare A matrices element-wise
    for (size_t i = 0; i < cfg.rank * in_dim; ++i) {
        REQUIRE(loaded.A_data()[i] == orig.A_data()[i]);
    }

    // Compare B matrices element-wise
    for (size_t i = 0; i < out_dim * cfg.rank; ++i) {
        REQUIRE(loaded.B_data()[i] == orig.B_data()[i]);
    }

    std::filesystem::remove_all(dir);
}

TEST_CASE("LoRA param_count is rank * (input_dim + output_dim)", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 16;

    nos::LoRAAdapter adapter(cfg, 256, 512);
    REQUIRE(adapter.param_count() == 16 * (256 + 512));

    nos::LoRAAdapter adapter2(cfg, 4096, 4096);
    REQUIRE(adapter2.param_count() == 16 * (4096 + 4096));
}

TEST_CASE("LoRA Kaiming He init stddev", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 32;

    const size_t in_dim = 1024;
    const size_t out_dim = 1024;
    nos::LoRAAdapter adapter(cfg, in_dim, out_dim);

    // Compute mean and stddev of A
    const float* A = adapter.A_data();
    const size_t n = cfg.rank * in_dim;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(A[i]);
    }
    double mean = sum / static_cast<double>(n);

    double var_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(A[i]) - mean;
        var_sum += diff * diff;
    }
    double stddev = std::sqrt(var_sum / static_cast<double>(n));

    // Expected stddev: sqrt(2/input_dim) = sqrt(2/1024) ~= 0.0442
    double expected = std::sqrt(2.0 / static_cast<double>(in_dim));

    // Statistical test: within 20% tolerance
    REQUIRE(std::fabs(stddev - expected) < 0.2 * expected);
    // Mean should be near zero
    REQUIRE(std::fabs(mean) < 0.01);
}

TEST_CASE("LoRA forward after non-zero B", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 4;
    cfg.alpha = 4.0f;

    const size_t in_dim = 8;
    const size_t out_dim = 8;
    nos::LoRAAdapter adapter(cfg, in_dim, out_dim);

    // Set B to non-zero
    float* B = adapter.B_data();
    for (size_t i = 0; i < out_dim * cfg.rank; ++i) {
        B[i] = 0.1f;
    }

    std::vector<float> x(in_dim, 1.0f);
    std::vector<float> delta(out_dim, 0.0f);

    adapter.forward(x.data(), delta.data());

    // Delta should now be non-zero
    bool has_nonzero = false;
    for (size_t i = 0; i < out_dim; ++i) {
        if (std::fabs(delta[i]) > 1e-8f) {
            has_nonzero = true;
            break;
        }
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("LoRA load validates dimensions", "[lora]") {
    nos::LoRAConfig cfg;
    cfg.rank = 4;
    cfg.alpha = 4.0f;

    nos::LoRAAdapter orig(cfg, 16, 16);

    auto dir = std::filesystem::temp_directory_path() / "neuralos_test" / "lora_dim_check";
    std::filesystem::remove_all(dir);
    REQUIRE(orig.save(dir.string()));

    // Try to load into an adapter with different dimensions
    nos::LoRAAdapter mismatch(cfg, 32, 32);
    REQUIRE_FALSE(mismatch.load(dir.string()));

    // Load into a fresh adapter (no pre-set dims) should succeed
    nos::LoRAAdapter fresh;
    REQUIRE(fresh.load(dir.string()));

    std::filesystem::remove_all(dir);
}
