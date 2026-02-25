/// @file bitnet_avx512.cpp
/// @brief AVX-512 implementation of BitNet b1.58 ternary matrix-vector multiply.
///
/// Compiled with -mavx512f -mavx512bw flags on this translation unit only.
/// Processes 32 INT16 elements per iteration (512 bits / 16 bits).
///
/// Integer accumulation in the inner loop, per-channel FP16 scale applied
/// ONCE per row OUTSIDE the inner loop. No floating-point in the hot path.

#include "bitnet_kernel.h"
#include "packing.h"

#include <immintrin.h>
#include <cstring>
#include <vector>

namespace nos {

void bitnet_matvec_avx512(const uint8_t* packed_weights, const float* input,
                          float* output, int rows, int cols,
                          const uint16_t* scale_factors) {
    const int bytes_per_row = (cols + 4) / 5;

    // Pre-unpack all trits for the matrix into a contiguous buffer.
    // This separates decode from compute so the SIMD inner loop is pure add/sub.
    std::vector<int8_t> all_trits(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; r++) {
        const uint8_t* row_packed = packed_weights + r * bytes_per_row;
        int8_t* row_trits = all_trits.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
        int trit_idx = 0;
        for (int b = 0; b < bytes_per_row && trit_idx < cols; b++) {
            int8_t tmp[5];
            unpack_5trits(row_packed[b], tmp);
            for (int t = 0; t < 5 && trit_idx < cols; t++, trit_idx++) {
                row_trits[trit_idx] = tmp[t];
            }
        }
    }

    // Pre-convert input to INT16 (scale by 128, truncate).
    // Done once, reused across all rows.
    std::vector<int16_t> input_i16(static_cast<size_t>(cols));
    for (size_t c = 0; c < static_cast<size_t>(cols); c++) {
        int32_t v = static_cast<int32_t>(input[c] * 128.0f);
        // Clamp to INT16 range
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        input_i16[c] = static_cast<int16_t>(v);
    }

    for (int r = 0; r < rows; r++) {
        const int8_t* row_trits = all_trits.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);

        // 32-bit accumulator vector (16 x int32)
        __m512i acc_lo = _mm512_setzero_si512();  // accumulates lower 16 elements
        __m512i acc_hi = _mm512_setzero_si512();  // accumulates upper 16 elements

        int col = 0;

        // Process 32 columns per iteration
        for (; col + 31 < cols; col += 32) {
            // Load 32 trits as int8, sign-extend to int16
            __m256i trits_i8 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row_trits + col));
            __m512i trits_i16 = _mm512_cvtepi8_epi16(trits_i8);

            // Load 32 pre-converted INT16 input values
            __m512i in_i16 = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(input_i16.data() + col));

            // Create masks: where trit == +1 and where trit == -1
            __m512i ones = _mm512_set1_epi16(1);
            __m512i neg_ones = _mm512_set1_epi16(-1);

            __mmask32 pos_mask = _mm512_cmpeq_epi16_mask(trits_i16, ones);
            __mmask32 neg_mask = _mm512_cmpeq_epi16_mask(trits_i16, neg_ones);

            // Widen input to int32 for accumulation (avoid INT16 overflow)
            __m512i in_lo = _mm512_cvtepi16_epi32(
                _mm512_castsi512_si256(in_i16));
            __m512i in_hi = _mm512_cvtepi16_epi32(
                _mm512_extracti64x4_epi64(in_i16, 1));

            // Split masks into lo/hi halves (16 bits each)
            __mmask16 pos_lo = static_cast<__mmask16>(pos_mask & 0xFFFF);
            __mmask16 pos_hi = static_cast<__mmask16>((pos_mask >> 16) & 0xFFFF);
            __mmask16 neg_lo = static_cast<__mmask16>(neg_mask & 0xFFFF);
            __mmask16 neg_hi = static_cast<__mmask16>((neg_mask >> 16) & 0xFFFF);

            // Masked add for +1 trits, masked sub for -1 trits (int32 accumulation)
            acc_lo = _mm512_mask_add_epi32(acc_lo, pos_lo, acc_lo, in_lo);
            acc_hi = _mm512_mask_add_epi32(acc_hi, pos_hi, acc_hi, in_hi);
            acc_lo = _mm512_mask_sub_epi32(acc_lo, neg_lo, acc_lo, in_lo);
            acc_hi = _mm512_mask_sub_epi32(acc_hi, neg_hi, acc_hi, in_hi);
        }

        // Handle tail (cols not multiple of 32)
        if (col < cols) {
            int remaining = cols - col;

            // Use masked load for tail elements
            __mmask32 tail_mask = (remaining >= 32) ? 0xFFFFFFFF
                : (static_cast<uint32_t>(1) << remaining) - 1;

            // Load remaining trits (zero-masked)
            __m256i trits_i8;
            if (remaining >= 32) {
                trits_i8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row_trits + col));
            } else {
                // Partial load: copy to stack buffer, zero-padded
                int8_t buf[32] = {};
                std::memcpy(buf, row_trits + col, static_cast<size_t>(remaining));
                trits_i8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf));
            }
            __m512i trits_i16 = _mm512_cvtepi8_epi16(trits_i8);

            // Load remaining input values (zero-padded)
            int16_t in_buf[32] = {};
            std::memcpy(in_buf, input_i16.data() + col,
                        static_cast<size_t>(remaining) * sizeof(int16_t));
            __m512i in_i16 = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(in_buf));

            __m512i ones = _mm512_set1_epi16(1);
            __m512i neg_ones = _mm512_set1_epi16(-1);

            // Mask comparisons AND with tail_mask to ignore padding
            __mmask32 pos_mask = _mm512_cmpeq_epi16_mask(trits_i16, ones) & tail_mask;
            __mmask32 neg_mask = _mm512_cmpeq_epi16_mask(trits_i16, neg_ones) & tail_mask;

            __m512i in_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(in_i16));
            __m512i in_hi = _mm512_cvtepi16_epi32(
                _mm512_extracti64x4_epi64(in_i16, 1));

            __mmask16 pos_lo = static_cast<__mmask16>(pos_mask & 0xFFFF);
            __mmask16 pos_hi = static_cast<__mmask16>((pos_mask >> 16) & 0xFFFF);
            __mmask16 neg_lo = static_cast<__mmask16>(neg_mask & 0xFFFF);
            __mmask16 neg_hi = static_cast<__mmask16>((neg_mask >> 16) & 0xFFFF);

            acc_lo = _mm512_mask_add_epi32(acc_lo, pos_lo, acc_lo, in_lo);
            acc_hi = _mm512_mask_add_epi32(acc_hi, pos_hi, acc_hi, in_hi);
            acc_lo = _mm512_mask_sub_epi32(acc_lo, neg_lo, acc_lo, in_lo);
            acc_hi = _mm512_mask_sub_epi32(acc_hi, neg_hi, acc_hi, in_hi);
        }

        // Horizontal sum: combine lo and hi accumulators, then reduce
        __m512i total = _mm512_add_epi32(acc_lo, acc_hi);
        int32_t acc = _mm512_reduce_add_epi32(total);

        // Per-channel scale applied ONCE per row, OUTSIDE the inner loop
        float scale = fp16_to_fp32(scale_factors[r]);
        output[r] = static_cast<float>(acc) / 128.0f * scale;
    }
}

}  // namespace nos
