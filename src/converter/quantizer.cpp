/// @file quantizer.cpp
/// @brief Ternary and INT8 quantization implementation.

#include "converter/quantizer.h"
#include "kernel/packing.h"  // pack_row, fp16_to_fp32, fp32_to_fp16

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace nos {

QuantizedWeights ternary_quantize(const uint16_t* fp16_weights, int rows, int cols) {
    QuantizedWeights result;
    result.rows = rows;
    result.cols = cols;

    int packed_cols = (cols + 4) / 5;  // 5 trits per byte
    result.packed.resize(static_cast<size_t>(rows) * static_cast<size_t>(packed_cols));
    result.scales.resize(static_cast<size_t>(rows));

    std::vector<int8_t> trits(static_cast<size_t>(cols));

    for (int r = 0; r < rows; r++) {
        const uint16_t* row = fp16_weights + r * cols;

        // Compute absmax for this channel (used for thresholding)
        float absmax = 0.0f;
        for (int c = 0; c < cols; c++) {
            float val = std::abs(fp16_to_fp32(row[c]));
            if (val > absmax) absmax = val;
        }

        // Quantize: threshold at 0.5 * absmax
        float threshold = 0.5f * absmax;

        for (int c = 0; c < cols; c++) {
            float val = fp16_to_fp32(row[c]);
            if (std::abs(val) > threshold) {
                trits[static_cast<size_t>(c)] = (val > 0.0f) ? 1 : -1;
            } else {
                trits[static_cast<size_t>(c)] = 0;
            }
        }

        // Compute scale as mean(|w|) of non-zero-quantized weights (BitNet b1.58)
        float sum_abs = 0.0f;
        int count = 0;
        for (int c = 0; c < cols; c++) {
            if (trits[static_cast<size_t>(c)] != 0) {
                sum_abs += std::abs(fp16_to_fp32(row[c]));
                count++;
            }
        }
        float alpha = (count > 0) ? (sum_abs / static_cast<float>(count)) : 0.0f;
        result.scales[static_cast<size_t>(r)] = fp32_to_fp16(alpha);

        // Pack trits using existing pack_row
        uint8_t* packed_row = result.packed.data()
                            + static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        pack_row(trits.data(), cols, packed_row);
    }

    return result;
}

Int8Weights int8_quantize(const uint16_t* fp16_weights, int rows, int cols) {
    Int8Weights result;
    result.rows = rows;
    result.cols = cols;
    result.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    result.scales.resize(static_cast<size_t>(rows));

    for (int r = 0; r < rows; r++) {
        const uint16_t* row = fp16_weights + r * cols;

        // Compute absmax
        float absmax = 0.0f;
        for (int c = 0; c < cols; c++) {
            float val = std::abs(fp16_to_fp32(row[c]));
            if (val > absmax) absmax = val;
        }

        // Scale: absmax / 127.0
        float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        result.scales[static_cast<size_t>(r)] = fp32_to_fp16(scale);

        // Quantize
        int8_t* out_row = result.data.data() + r * cols;
        for (int c = 0; c < cols; c++) {
            float val = fp16_to_fp32(row[c]);
            float q = std::round(val / scale);
            q = std::max(-127.0f, std::min(127.0f, q));
            out_row[c] = static_cast<int8_t>(q);
        }
    }

    return result;
}

}  // namespace nos
