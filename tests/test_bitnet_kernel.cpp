/// @file test_bitnet_kernel.cpp
/// @brief Correctness tests for BitNet ternary kernels: scalar golden reference
///        and cross-ISA comparison.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "bitnet_kernel.h"
#include "packing.h"

using namespace nos;
using Catch::Approx;

// ── Test Helpers ─────────────────────────────────────────────────────────────

/// Generate random ternary weights {-1, 0, +1} and pack them.
static void generate_random_packed_weights(int rows, int cols,
                                            std::vector<uint8_t>& packed,
                                            std::vector<int8_t>& trits,
                                            std::mt19937& rng) {
    trits.resize(rows * cols);
    std::uniform_int_distribution<int> dist(-1, 1);
    for (auto& t : trits) {
        t = static_cast<int8_t>(dist(rng));
    }

    int bytes_per_row = (cols + 4) / 5;
    packed.resize(rows * bytes_per_row);
    for (int r = 0; r < rows; r++) {
        pack_row(trits.data() + r * cols, cols, packed.data() + r * bytes_per_row);
    }
}

/// Generate random float input vector.
static void generate_random_input(int cols, std::vector<float>& input, std::mt19937& rng) {
    input.resize(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : input) {
        v = dist(rng);
    }
}

/// Generate random FP16 scale factors (positive, reasonable range).
static void generate_random_scales(int rows, std::vector<uint16_t>& scales, std::mt19937& rng) {
    scales.resize(rows);
    std::uniform_real_distribution<float> dist(0.001f, 2.0f);
    for (auto& s : scales) {
        s = fp32_to_fp16(dist(rng));
    }
}

/// Compute reference matmul in plain float (no packing) for known-output tests.
static void reference_matvec_float(const int8_t* trits, const float* input,
                                    float* output, int rows, int cols,
                                    const uint16_t* scales) {
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) {
            acc += static_cast<float>(trits[r * cols + c]) * input[c];
        }
        output[r] = acc * fp16_to_fp32(scales[r]);
    }
}

// ── Scalar Kernel Tests ──────────────────────────────────────────────────────

TEST_CASE("Scalar kernel: known 4x10 matrix", "[bitnet_kernel][scalar]") {
    // 4 rows, 10 cols -- manually defined
    const int rows = 4;
    const int cols = 10;

    // Known trit matrix
    int8_t trits[] = {
        // Row 0: all +1
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        // Row 1: all -1
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        // Row 2: alternating
         1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
        // Row 3: zeros with some non-zeros
         0,  0,  1,  0,  0, -1,  0,  0,  0,  0,
    };

    // Known input: 1.0f for all elements
    float input[cols];
    for (int i = 0; i < cols; i++) input[i] = 1.0f;

    // Known scales: all 1.0 in FP16
    uint16_t scales[rows];
    for (int r = 0; r < rows; r++) scales[r] = fp32_to_fp16(1.0f);

    // Pack weights
    int bytes_per_row = (cols + 4) / 5;
    std::vector<uint8_t> packed(rows * bytes_per_row);
    for (int r = 0; r < rows; r++) {
        pack_row(trits + r * cols, cols, packed.data() + r * bytes_per_row);
    }

    float output[rows] = {};
    bitnet_matvec_scalar(packed.data(), input, output, rows, cols, scales);

    // Expected:
    // Row 0: sum of 10 * 1.0 = 10.0
    // Row 1: sum of 10 * -1.0 = -10.0
    // Row 2: 5*1.0 + 5*(-1.0) = 0.0
    // Row 3: 1.0 + (-1.0) = 0.0
    // Note: scalar kernel quantizes input to int via * 128 then / 128, so
    // exact values should be very close

    // Reference computation
    float ref_output[rows];
    reference_matvec_float(trits, input, ref_output, rows, cols, scales);

    for (int r = 0; r < rows; r++) {
        REQUIRE(output[r] == Approx(ref_output[r]).epsilon(1e-4).margin(1e-5));
    }
}

TEST_CASE("Scalar kernel: random matrix, no NaN/Inf", "[bitnet_kernel][scalar]") {
    const int rows = 128;
    const int cols = 256;
    std::mt19937 rng(42);

    std::vector<uint8_t> packed;
    std::vector<int8_t> trits;
    generate_random_packed_weights(rows, cols, packed, trits, rng);

    std::vector<float> input;
    generate_random_input(cols, input, rng);

    std::vector<uint16_t> scales;
    generate_random_scales(rows, scales, rng);

    std::vector<float> output(rows);
    bitnet_matvec_scalar(packed.data(), input.data(), output.data(),
                          rows, cols, scales.data());

    for (int r = 0; r < rows; r++) {
        REQUIRE_FALSE(std::isnan(output[r]));
        REQUIRE_FALSE(std::isinf(output[r]));
    }
}

TEST_CASE("Scalar kernel: dimension sweep", "[bitnet_kernel][scalar]") {
    std::mt19937 rng(123);

    auto test_dims = [&](int rows, int cols) {
        INFO("rows=" << rows << " cols=" << cols);

        std::vector<uint8_t> packed;
        std::vector<int8_t> trits;
        generate_random_packed_weights(rows, cols, packed, trits, rng);

        std::vector<float> input;
        generate_random_input(cols, input, rng);

        std::vector<uint16_t> scales;
        generate_random_scales(rows, scales, rng);

        // Compute with scalar kernel
        std::vector<float> output(rows);
        bitnet_matvec_scalar(packed.data(), input.data(), output.data(),
                              rows, cols, scales.data());

        // Compute reference (no packing)
        std::vector<float> ref(rows);
        reference_matvec_float(trits.data(), input.data(), ref.data(),
                                rows, cols, scales.data());

        for (int r = 0; r < rows; r++) {
            REQUIRE_FALSE(std::isnan(output[r]));
            REQUIRE_FALSE(std::isinf(output[r]));
            // Tolerance: scalar kernel quantizes input*128 to int32, introducing
            // ~1/128 rounding error per element. Over cols elements, the accumulated
            // error is proportional to cols. Use 2% relative + absolute margin.
            float abs_margin = static_cast<float>(cols) * 0.005f;
            REQUIRE(output[r] == Approx(ref[r]).epsilon(0.02).margin(abs_margin));
        }
    };

    SECTION("cols=1") { test_dims(2, 1); }
    SECTION("cols=5") { test_dims(2, 5); }
    SECTION("cols=7") { test_dims(2, 7); }
    SECTION("cols=32") { test_dims(4, 32); }
    SECTION("cols=127") { test_dims(4, 127); }
    SECTION("cols=256") { test_dims(4, 256); }
    SECTION("cols=4096") { test_dims(4, 4096); }
    SECTION("cols=4097") { test_dims(4, 4097); }
    // Llama-realistic dimensions
    SECTION("cols=11008") { test_dims(2, 11008); }
}

TEST_CASE("Scalar kernel: scale factors applied correctly", "[bitnet_kernel][scalar]") {
    const int rows = 4;
    const int cols = 5;

    // All trits = +1, input = 1.0f
    int8_t trits[rows * cols];
    std::fill(trits, trits + rows * cols, static_cast<int8_t>(1));

    float input[cols];
    std::fill(input, input + cols, 1.0f);

    int bytes_per_row = (cols + 4) / 5;
    std::vector<uint8_t> packed(rows * bytes_per_row);
    for (int r = 0; r < rows; r++) {
        pack_row(trits + r * cols, cols, packed.data() + r * bytes_per_row);
    }

    // Different scales for each row
    float scale_vals[] = {0.5f, 1.0f, 2.0f, 0.25f};
    uint16_t scales[rows];
    for (int r = 0; r < rows; r++) scales[r] = fp32_to_fp16(scale_vals[r]);

    float output[rows] = {};
    bitnet_matvec_scalar(packed.data(), input, output, rows, cols, scales);

    // Each row should be: 5.0 * scale[r] (sum of 5 ones * scale)
    for (int r = 0; r < rows; r++) {
        float expected = 5.0f * scale_vals[r];
        REQUIRE(output[r] == Approx(expected).epsilon(1e-3));
    }
}

// ── Dispatch Tests ───────────────────────────────────────────────────────────

TEST_CASE("bitnet_init selects a backend", "[bitnet_kernel][dispatch]") {
    bitnet_init();
    const char* name = bitnet_get_backend_name();
    REQUIRE(name != nullptr);
    // Must be one of the known backends
    bool valid = (strcmp(name, "scalar") == 0) ||
                 (strcmp(name, "avx512") == 0) ||
                 (strcmp(name, "neon") == 0);
    REQUIRE(valid);
    INFO("Selected backend: " << name);
}

TEST_CASE("bitnet_matvec dispatches correctly", "[bitnet_kernel][dispatch]") {
    bitnet_init();

    const int rows = 4;
    const int cols = 10;
    std::mt19937 rng(99);

    std::vector<uint8_t> packed;
    std::vector<int8_t> trits;
    generate_random_packed_weights(rows, cols, packed, trits, rng);

    std::vector<float> input;
    generate_random_input(cols, input, rng);

    std::vector<uint16_t> scales;
    generate_random_scales(rows, scales, rng);

    // Compute via dispatch
    std::vector<float> output(rows);
    bitnet_matvec(packed.data(), input.data(), output.data(),
                   rows, cols, scales.data());

    // Compute via scalar directly
    std::vector<float> scalar_output(rows);
    bitnet_matvec_scalar(packed.data(), input.data(), scalar_output.data(),
                          rows, cols, scales.data());

    // Should match (either using scalar or SIMD)
    for (int r = 0; r < rows; r++) {
        REQUIRE(output[r] == Approx(scalar_output[r]).epsilon(1e-5).margin(1e-6));
    }
}
