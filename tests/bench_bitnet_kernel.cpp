/// @file bench_bitnet_kernel.cpp
/// @brief Benchmark BitNet ternary matmul kernels.
///
/// Measures scalar (and SIMD if available) throughput on realistic dimensions.
/// Reports GFLOP/s equivalent and scalar vs SIMD speedup.

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bitnet_kernel.h"
#include "packing.h"

using namespace nos;

// ── Scalar Matmul Benchmark ─────────────────────────────────────────────────

TEST_CASE("Benchmark: scalar matmul 4096x4096", "[.benchmark][bitnet_kernel]") {
    const int rows = 4096;
    const int cols = 4096;
    const int bytes_per_row = (cols + 4) / 5;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit_dist(-1, 1);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);

    // Generate packed weights
    std::vector<int8_t> trits(rows * cols);
    for (auto& t : trits) t = static_cast<int8_t>(trit_dist(rng));

    std::vector<uint8_t> packed(rows * bytes_per_row);
    for (int r = 0; r < rows; r++) {
        pack_row(trits.data() + r * cols, cols, packed.data() + r * bytes_per_row);
    }

    std::vector<float> input(cols);
    for (auto& v : input) v = float_dist(rng);

    std::vector<uint16_t> scales(rows);
    for (auto& s : scales) s = fp32_to_fp16(std::abs(float_dist(rng)) + 0.01f);

    std::vector<float> output(rows);

    BENCHMARK("scalar 4096x4096") {
        bitnet_matvec_scalar(packed.data(), input.data(), output.data(),
                              rows, cols, scales.data());
        return output[0];
    };
}

// ── Dispatched Matmul Benchmark ─────────────────────────────────────────────

TEST_CASE("Benchmark: dispatched matmul 4096x4096", "[.benchmark][bitnet_kernel]") {
    bitnet_init();
    INFO("Backend: " << bitnet_get_backend_name());

    const int rows = 4096;
    const int cols = 4096;
    const int bytes_per_row = (cols + 4) / 5;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit_dist(-1, 1);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);

    std::vector<int8_t> trits(rows * cols);
    for (auto& t : trits) t = static_cast<int8_t>(trit_dist(rng));

    std::vector<uint8_t> packed(rows * bytes_per_row);
    for (int r = 0; r < rows; r++) {
        pack_row(trits.data() + r * cols, cols, packed.data() + r * bytes_per_row);
    }

    std::vector<float> input(cols);
    for (auto& v : input) v = float_dist(rng);

    std::vector<uint16_t> scales(rows);
    for (auto& s : scales) s = fp32_to_fp16(std::abs(float_dist(rng)) + 0.01f);

    std::vector<float> output(rows);

    BENCHMARK("dispatched 4096x4096") {
        bitnet_matvec(packed.data(), input.data(), output.data(),
                       rows, cols, scales.data());
        return output[0];
    };
}

// ── Llama-realistic Dimensions ──────────────────────────────────────────────

TEST_CASE("Benchmark: scalar matmul Llama dims", "[.benchmark][bitnet_kernel]") {
    const int rows = 4096;
    const int cols = 11008;  // Llama 7B intermediate dim
    const int bytes_per_row = (cols + 4) / 5;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit_dist(-1, 1);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);

    std::vector<int8_t> trits(rows * cols);
    for (auto& t : trits) t = static_cast<int8_t>(trit_dist(rng));

    std::vector<uint8_t> packed(rows * bytes_per_row);
    for (int r = 0; r < rows; r++) {
        pack_row(trits.data() + r * cols, cols, packed.data() + r * bytes_per_row);
    }

    std::vector<float> input(cols);
    for (auto& v : input) v = float_dist(rng);

    std::vector<uint16_t> scales(rows);
    for (auto& s : scales) s = fp32_to_fp16(std::abs(float_dist(rng)) + 0.01f);

    std::vector<float> output(rows);

    BENCHMARK("scalar 4096x11008 (Llama)") {
        bitnet_matvec_scalar(packed.data(), input.data(), output.data(),
                              rows, cols, scales.data());
        return output[0];
    };
}
