/// @file bitnet_avx512.cpp
/// @brief AVX-512 implementation of BitNet b1.58 ternary matrix-vector multiply.
///
/// Stub -- full implementation in Task 2.
/// Compiled with -mavx512f -mavx512bw flags on this translation unit only.

#include "bitnet_kernel.h"
#include "packing.h"

namespace nos {

// Stub: delegates to scalar for now. Full AVX-512 implementation in Task 2.
void bitnet_matvec_avx512(const uint8_t* packed_weights, const float* input, float* output,
                          int rows, int cols, const uint16_t* scale_factors) {
    bitnet_matvec_scalar(packed_weights, input, output, rows, cols, scale_factors);
}

}  // namespace nos
