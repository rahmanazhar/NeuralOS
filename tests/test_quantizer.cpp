/// @file test_quantizer.cpp
/// @brief Tests for ternary and INT8 quantization.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "converter/quantizer.h"
#include "kernel/packing.h"  // fp32_to_fp16, fp16_to_fp32, unpack_row

using Catch::Matchers::WithinAbs;

/// Create a known FP16 weight matrix from FP32 values.
static std::vector<uint16_t> make_fp16(const std::vector<float>& f32) {
    std::vector<uint16_t> fp16(f32.size());
    for (size_t i = 0; i < f32.size(); i++) {
        fp16[i] = nos::fp32_to_fp16(f32[i]);
    }
    return fp16;
}

TEST_CASE("Ternary quantization: basic correctness", "[quantizer]") {
    // 2x8 weight matrix with known values
    // Row 0: values with clear ternary pattern
    //   absmax = 2.0, threshold = 1.0
    //   Values: 2.0, -1.5, 0.3, -0.2, 1.8, -2.0, 0.1, 0.9
    //   Trits:   +1,   -1,   0,    0,  +1,   -1,   0,   0
    // Row 1: all zeros -> all trits should be 0
    //   absmax = 0, threshold = 0
    std::vector<float> weights = {
        2.0f, -1.5f, 0.3f, -0.2f, 1.8f, -2.0f, 0.1f, 0.9f,  // row 0
        0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f, 0.0f   // row 1
    };
    auto fp16 = make_fp16(weights);

    auto result = nos::ternary_quantize(fp16.data(), 2, 8);

    CHECK(result.rows == 2);
    CHECK(result.cols == 8);

    SECTION("packed size is correct") {
        int packed_cols = (8 + 4) / 5;  // = 2 bytes per row
        CHECK(result.packed.size() == static_cast<size_t>(2 * packed_cols));
    }

    SECTION("row 0 trits match expected") {
        int packed_cols = (8 + 4) / 5;
        std::vector<int8_t> trits(8);
        nos::unpack_row(result.packed.data(), 8, trits.data());

        CHECK(trits[0] ==  1);   // 2.0 > 1.0
        CHECK(trits[1] == -1);   // |-1.5| > 1.0
        CHECK(trits[2] ==  0);   // |0.3| <= 1.0
        CHECK(trits[3] ==  0);   // |0.2| <= 1.0
        CHECK(trits[4] ==  1);   // 1.8 > 1.0
        CHECK(trits[5] == -1);   // |-2.0| > 1.0
        CHECK(trits[6] ==  0);   // |0.1| <= 1.0
        CHECK(trits[7] ==  0);   // |0.9| <= 1.0

        (void)packed_cols;
    }

    SECTION("row 0 scale is absmax") {
        float scale = nos::fp16_to_fp32(result.scales[0]);
        CHECK_THAT(static_cast<double>(scale), WithinAbs(2.0, 0.01));
    }

    SECTION("row 1 (all zeros) produces zero trits") {
        int packed_cols = (8 + 4) / 5;
        std::vector<int8_t> trits(8);
        nos::unpack_row(result.packed.data() + packed_cols, 8, trits.data());

        for (int i = 0; i < 8; i++) {
            CHECK(trits[static_cast<size_t>(i)] == 0);
        }
    }
}

TEST_CASE("Ternary quantization: round-trip reconstruction error", "[quantizer]") {
    // Create a weight matrix with known values and verify reconstruction
    int rows = 4;
    int cols = 10;
    std::vector<float> weights(static_cast<size_t>(rows * cols));
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // Alternate large and small values
            float val = ((r * cols + c) % 3 == 0) ? 1.5f : 0.2f;
            if ((r * cols + c) % 2 == 0) val = -val;
            weights[static_cast<size_t>(r * cols + c)] = val;
        }
    }
    auto fp16 = make_fp16(weights);

    auto result = nos::ternary_quantize(fp16.data(), rows, cols);

    // Dequantize and check reconstruction error
    int packed_cols = (cols + 4) / 5;
    for (int r = 0; r < rows; r++) {
        float scale = nos::fp16_to_fp32(result.scales[static_cast<size_t>(r)]);

        std::vector<int8_t> trits(static_cast<size_t>(cols));
        nos::unpack_row(result.packed.data() + r * packed_cols, cols, trits.data());

        for (int c = 0; c < cols; c++) {
            float reconstructed = static_cast<float>(trits[static_cast<size_t>(c)]) * scale;
            float original = weights[static_cast<size_t>(r * cols + c)];

            // Ternary quantization has significant error, but the sign should match
            // for large values and reconstruct to zero for small values
            if (std::abs(original) > 0.5f * scale) {
                // Should have correct sign
                CHECK(((original > 0.0f && reconstructed > 0.0f) ||
                       (original < 0.0f && reconstructed < 0.0f)));
            } else {
                CHECK(reconstructed == 0.0f);
            }
        }
    }
}

TEST_CASE("INT8 quantization: basic correctness", "[quantizer]") {
    // 2x4 weight matrix
    std::vector<float> weights = {
        1.0f, -0.5f, 0.25f, -1.0f,  // row 0: absmax = 1.0
        0.0f,  0.0f, 0.0f,   0.0f   // row 1: all zeros
    };
    auto fp16 = make_fp16(weights);

    auto result = nos::int8_quantize(fp16.data(), 2, 4);

    CHECK(result.rows == 2);
    CHECK(result.cols == 4);
    CHECK(result.data.size() == 8);
    CHECK(result.scales.size() == 2);

    SECTION("row 0 quantized values") {
        float scale = nos::fp16_to_fp32(result.scales[0]);
        CHECK_THAT(static_cast<double>(scale), WithinAbs(1.0 / 127.0, 1e-3));

        // 1.0 / scale = 127
        CHECK(result.data[0] == 127);
        // -0.5 / scale ~ -63.5 -> -64 (rounded)
        CHECK(std::abs(result.data[1] - (-64)) <= 1);
        // 0.25 / scale ~ 31.75 -> 32
        CHECK(std::abs(result.data[2] - 32) <= 1);
        // -1.0 / scale = -127
        CHECK(result.data[3] == -127);
    }

    SECTION("row 1 all zeros") {
        for (int c = 0; c < 4; c++) {
            CHECK(result.data[4 + c] == 0);
        }
    }
}

TEST_CASE("INT8 quantization: round-trip reconstruction error", "[quantizer]") {
    int rows = 4;
    int cols = 8;
    std::vector<float> weights(static_cast<size_t>(rows * cols));
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = static_cast<float>(i) * 0.1f - 1.5f;
    }
    auto fp16 = make_fp16(weights);

    auto result = nos::int8_quantize(fp16.data(), rows, cols);

    // Dequantize and verify error is small
    for (int r = 0; r < rows; r++) {
        float scale = nos::fp16_to_fp32(result.scales[static_cast<size_t>(r)]);

        for (int c = 0; c < cols; c++) {
            float original = weights[static_cast<size_t>(r * cols + c)];
            float reconstructed = static_cast<float>(result.data[r * cols + c]) * scale;
            float error = std::abs(original - reconstructed);

            // INT8 max error per value is 0.5 * scale (rounding)
            CHECK(error < scale + 0.01f);
        }
    }
}

TEST_CASE("INT8 quantization: clamping to [-127, 127]", "[quantizer]") {
    // Values that should hit the clamp limits
    std::vector<float> weights = {
        10.0f, -10.0f, 5.0f, -5.0f
    };
    auto fp16 = make_fp16(weights);

    auto result = nos::int8_quantize(fp16.data(), 1, 4);

    // absmax = 10.0, scale = 10.0/127.0
    // 10.0 / scale = 127, -10.0 / scale = -127
    CHECK(result.data[0] == 127);
    CHECK(result.data[1] == -127);
}
