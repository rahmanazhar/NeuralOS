#pragma once

/// @file quantizer.h
/// @brief Ternary and INT8 quantization for expert and attention weights.
///
/// Ternary quantization produces {-1, 0, +1} with per-channel FP16 scales
/// using absmax threshold at 0.5*scale. INT8 quantization uses per-channel
/// absmax scaling for attention weights.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nos {

/// Packed ternary weight matrix with per-channel FP16 scales.
struct QuantizedWeights {
    std::vector<uint8_t> packed;    ///< Packed trits (5-per-byte encoding)
    std::vector<uint16_t> scales;   ///< Per-channel FP16 scale factors
    int rows = 0;
    int cols = 0;
};

/// INT8 weight matrix with per-channel FP16 scales.
struct Int8Weights {
    std::vector<int8_t> data;       ///< INT8 quantized weights
    std::vector<uint16_t> scales;   ///< Per-channel FP16 scale factors
    int rows = 0;
    int cols = 0;
};

/// Ternary quantize an FP16 weight matrix.
///
/// For each output channel (row): compute absmax, scale = absmax.
/// For each weight: if abs(w) > 0.5 * scale, trit = sign(w); else trit = 0.
/// Pack trits using pack_row() from packing.h.
///
/// @param fp16_weights  FP16 weight matrix [rows x cols], row-major
/// @param rows          Number of output channels
/// @param cols          Number of input features
/// @return Packed ternary weights with per-channel scales
QuantizedWeights ternary_quantize(const uint16_t* fp16_weights, int rows, int cols);

/// INT8 quantize an FP16 weight matrix with per-channel absmax scaling.
///
/// For each row: absmax -> scale = absmax / 127.0.
/// Quantize: int8_val = round(fp32_val / scale), clamped to [-127, 127].
///
/// @param fp16_weights  FP16 weight matrix [rows x cols], row-major
/// @param rows          Number of output channels
/// @param cols          Number of input features
/// @return INT8 weights with per-channel scales
Int8Weights int8_quantize(const uint16_t* fp16_weights, int rows, int cols);

}  // namespace nos
