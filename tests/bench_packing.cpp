/// @file bench_packing.cpp
/// @brief Benchmark 5-per-byte ternary packing decode overhead.
///
/// Measures decode throughput, encode throughput, and decode as a percentage
/// of a scalar matmul forward pass for a 128x4096 matrix.

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bitnet_kernel.h"
#include "packing.h"

using namespace nos;

// ── Decode Throughput ────────────────────────────────────────────────────────

TEST_CASE("Benchmark: decode throughput (1MB)", "[.benchmark][packing]") {
    // 1 MB of packed data
    const int num_bytes = 1024 * 1024;
    std::vector<uint8_t> packed(num_bytes);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 242);
    for (auto& b : packed) b = static_cast<uint8_t>(dist(rng));

    std::vector<int8_t> output(num_bytes * 5);

    BENCHMARK("unpack 1MB packed data") {
        for (int i = 0; i < num_bytes; i++) {
            unpack_5trits(packed[i], output.data() + i * 5);
        }
        return output[0];
    };
}

// ── Encode Throughput ────────────────────────────────────────────────────────

TEST_CASE("Benchmark: encode throughput (5M trits)", "[.benchmark][packing]") {
    // 5M trits -> 1M packed bytes
    const int num_trits = 5 * 1024 * 1024;
    std::vector<int8_t> trits(num_trits);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1, 1);
    for (auto& t : trits) t = static_cast<int8_t>(dist(rng));

    std::vector<uint8_t> packed(num_trits / 5);

    BENCHMARK("pack 5M trits") {
        pack_row(trits.data(), num_trits, packed.data());
        return packed[0];
    };
}

// ── Decode as % of Scalar Forward Pass ───────────────────────────────────────

TEST_CASE("Benchmark: decode overhead as % of scalar forward pass", "[.benchmark][packing]") {
    // Matrix: 128 rows x 4096 cols (realistic)
    const int rows = 128;
    const int cols = 4096;
    const int bytes_per_row = (cols + 4) / 5;
    const int total_bytes = rows * bytes_per_row;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> trit_dist(-1, 1);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);

    // Generate packed weights
    std::vector<int8_t> trits(rows * cols);
    for (auto& t : trits) t = static_cast<int8_t>(trit_dist(rng));

    std::vector<uint8_t> packed(total_bytes);
    for (int r = 0; r < rows; r++) {
        pack_row(trits.data() + r * cols, cols, packed.data() + r * bytes_per_row);
    }

    // Generate input and scales
    std::vector<float> input(cols);
    for (auto& v : input) v = float_dist(rng);

    std::vector<uint16_t> scales(rows);
    for (auto& s : scales) s = fp32_to_fp16(std::abs(float_dist(rng)) + 0.01f);

    std::vector<float> output(rows);

    // Benchmark decode only
    std::vector<int8_t> decode_buf(total_bytes * 5);
    BENCHMARK("decode only (128x4096)") {
        for (int i = 0; i < total_bytes; i++) {
            unpack_5trits(packed[i], decode_buf.data() + i * 5);
        }
        return decode_buf[0];
    };

    // Benchmark full scalar matmul (includes decode)
    BENCHMARK("scalar matmul (128x4096)") {
        bitnet_matvec_scalar(packed.data(), input.data(), output.data(),
                              rows, cols, scales.data());
        return output[0];
    };

    // Note: The ratio of decode/matmul gives the decode overhead percentage.
    // Printed in the Catch2 benchmark output for analysis.
}
