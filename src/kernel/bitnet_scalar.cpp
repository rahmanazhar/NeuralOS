/// @file bitnet_scalar.cpp
/// @brief Scalar reference implementation of BitNet b1.58 ternary matrix-vector multiply.
///
/// This is the GOLDEN REFERENCE. Correctness over performance.
/// All SIMD backends must produce output matching this within float tolerance.
///
/// Integer accumulation in the inner loop, per-channel FP16 scale applied
/// ONCE per row OUTSIDE the inner loop.

#include "bitnet_kernel.h"
#include "packing.h"

namespace nos {

void bitnet_matvec_scalar(const uint8_t* packed_w,
                          const float* input,
                          float* output,
                          int rows, int cols,
                          const uint16_t* scale_factors) {
    const int bytes_per_row = (cols + 4) / 5;
    int8_t trits[5];

    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        int col = 0;
        const uint8_t* row_packed = packed_w + r * bytes_per_row;

        for (int b = 0; b < bytes_per_row; b++) {
            unpack_5trits(row_packed[b], trits);
            for (int t = 0; t < 5 && col < cols; t++, col++) {
                if (trits[t] == 1) {
                    acc += static_cast<int32_t>(input[col] * 128.0f);
                } else if (trits[t] == -1) {
                    acc -= static_cast<int32_t>(input[col] * 128.0f);
                }
                // trit == 0: skip (add nothing)
            }
        }

        // Per-channel scale applied ONCE per row, OUTSIDE the inner loop
        float scale = fp16_to_fp32(scale_factors[r]);
        output[r] = static_cast<float>(acc) / 128.0f * scale;
    }
}

}  // namespace nos
