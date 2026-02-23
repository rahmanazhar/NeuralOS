/// @file bitnet_dispatch.cpp
/// @brief Placeholder for runtime ISA dispatch. Full implementation in Plan 02.

#include "bitnet_kernel.h"

namespace nos {

static MatVecFn g_matvec_fn = nullptr;

void bitnet_init() {
    // Stub: will be implemented with CPUID-based dispatch in Plan 02.
    g_matvec_fn = nullptr;
}

void bitnet_matvec(const uint8_t* packed_weights, const float* input, float* output, int rows,
                   int cols, const uint16_t* scale_factors) {
    if (g_matvec_fn) {
        g_matvec_fn(packed_weights, input, output, rows, cols, scale_factors);
    }
}

}  // namespace nos
