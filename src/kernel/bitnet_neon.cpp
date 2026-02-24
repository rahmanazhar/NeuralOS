/// @file bitnet_neon.cpp
/// @brief NEON implementation of BitNet b1.58 ternary matrix-vector multiply.
///
/// Compiled only on AArch64 where NEON is baseline.
/// Processes 8 INT16 elements per iteration (128 bits / 16 bits).

#include "bitnet_kernel.h"
#include "packing.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace nos {

void bitnet_matvec_neon(const uint8_t* packed_weights, const float* input,
                        float* output, int rows, int cols,
                        const uint16_t* scale_factors) {
#if defined(__aarch64__) || defined(_M_ARM64)
    const int bytes_per_row = (cols + 4) / 5;

    for (int r = 0; r < rows; r++) {
        int32x4_t acc_lo = vdupq_n_s32(0);
        int32x4_t acc_hi = vdupq_n_s32(0);

        const uint8_t* row_packed = packed_weights + r * bytes_per_row;

        // Unpack all trits for this row into a buffer
        // Process 8 columns at a time with NEON
        int col = 0;
        int8_t trit_buf[8];

        // Process in chunks of 8 columns
        int full_chunks = cols / 8;
        int trit_idx = 0;
        int byte_idx = 0;
        int trit_in_byte = 0;

        // Pre-unpack all trits for this row
        std::vector<int8_t> all_trits(cols);
        for (int b = 0; b < bytes_per_row && trit_idx < cols; b++) {
            int8_t tmp[5];
            unpack_5trits(row_packed[b], tmp);
            for (int t = 0; t < 5 && trit_idx < cols; t++, trit_idx++) {
                all_trits[trit_idx] = tmp[t];
            }
        }

        col = 0;
        for (; col + 7 < cols; col += 8) {
            // Load 8 trits as int8, widen to int16
            int8x8_t trits_i8 = vld1_s8(&all_trits[col]);
            int16x8_t trits_i16 = vmovl_s8(trits_i8);

            // Load 8 floats, convert to int16 (scale by 128)
            float32x4_t in_lo = vld1q_f32(&input[col]);
            float32x4_t in_hi = vld1q_f32(&input[col + 4]);

            float32x4_t scale128 = vdupq_n_f32(128.0f);
            int32x4_t in_i32_lo = vcvtq_s32_f32(vmulq_f32(in_lo, scale128));
            int32x4_t in_i32_hi = vcvtq_s32_f32(vmulq_f32(in_hi, scale128));

            int16x4_t in_i16_lo = vmovn_s32(in_i32_lo);
            int16x4_t in_i16_hi = vmovn_s32(in_i32_hi);
            int16x8_t in_i16 = vcombine_s16(in_i16_lo, in_i16_hi);

            // Create masks: +1 and -1
            int16x8_t ones = vdupq_n_s16(1);
            int16x8_t neg_ones = vdupq_n_s16(-1);

            uint16x8_t pos_mask = vceqq_s16(trits_i16, ones);
            uint16x8_t neg_mask = vceqq_s16(trits_i16, neg_ones);

            // Masked add for +1 trits, masked sub for -1 trits
            int16x8_t pos_vals = vandq_s16(vreinterpretq_s16_u16(pos_mask), in_i16);
            int16x8_t neg_vals = vandq_s16(vreinterpretq_s16_u16(neg_mask), in_i16);

            // Widen to int32 and accumulate
            acc_lo = vaddq_s32(acc_lo, vmovl_s16(vget_low_s16(pos_vals)));
            acc_hi = vaddq_s32(acc_hi, vmovl_s16(vget_high_s16(pos_vals)));
            acc_lo = vsubq_s32(acc_lo, vmovl_s16(vget_low_s16(neg_vals)));
            acc_hi = vsubq_s32(acc_hi, vmovl_s16(vget_high_s16(neg_vals)));
        }

        // Horizontal sum
        int32x4_t sum = vaddq_s32(acc_lo, acc_hi);
        int32_t acc = vaddvq_s32(sum);

        // Scalar tail for remaining columns
        for (; col < cols; col++) {
            int8_t trit = all_trits[col];
            if (trit == 1) {
                acc += static_cast<int32_t>(input[col] * 128.0f);
            } else if (trit == -1) {
                acc -= static_cast<int32_t>(input[col] * 128.0f);
            }
        }

        // Per-channel scale applied ONCE per row, OUTSIDE the inner loop
        float scale = fp16_to_fp32(scale_factors[r]);
        output[r] = static_cast<float>(acc) / 128.0f * scale;
    }
#else
    // Fallback to scalar on non-AArch64 (should never be called)
    bitnet_matvec_scalar(packed_weights, input, output, rows, cols, scale_factors);
#endif
}

}  // namespace nos
