#pragma once

/// @file bitnet_kernel.h
/// @brief BitNet b1.58 ternary compute kernel public interface.
///
/// The kernel performs ternary matrix-vector multiplication using integer
/// add/sub (no FMA). Per-channel FP16 scale factors are applied once per
/// output row, outside the SIMD inner loop.

#include <cstddef>
#include <cstdint>

namespace nos {

/// Function pointer type for SIMD-dispatched ternary matrix-vector multiply.
///
/// @param packed_weights  Packed ternary weights (5-per-byte encoding)
/// @param input           Input activation vector (FP32)
/// @param output          Output vector (FP32), caller-allocated
/// @param rows            Number of output rows (hidden_dim)
/// @param cols            Number of input columns (intermediate_dim)
/// @param scale_factors   Per-channel FP16 scale factors, one per row
using MatVecFn = void (*)(const uint8_t* packed_weights, const float* input, float* output,
                          int rows, int cols, const uint16_t* scale_factors);

/// Initialize the kernel dispatch table.
///
/// Detects ISA capabilities at runtime (CPUID on x86, baseline on AArch64)
/// and selects the fastest available kernel implementation.
/// Must be called before bitnet_matvec().
void bitnet_init();

/// Perform ternary matrix-vector multiplication.
///
/// Dispatches to the ISA-specific implementation selected by bitnet_init().
/// The inner loop accumulates in integer; scale factors are applied per-row.
///
/// @param packed_weights  Packed ternary weights (5-per-byte)
/// @param input           Input activation vector (FP32)
/// @param output          Output vector (FP32), must be pre-allocated with rows elements
/// @param rows            Number of output rows
/// @param cols            Number of input columns
/// @param scale_factors   Per-channel FP16 scale factors (stored as uint16_t)
void bitnet_matvec(const uint8_t* packed_weights, const float* input, float* output, int rows,
                   int cols, const uint16_t* scale_factors);

}  // namespace nos
