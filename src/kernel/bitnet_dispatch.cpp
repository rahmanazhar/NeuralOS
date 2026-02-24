/// @file bitnet_dispatch.cpp
/// @brief Runtime ISA dispatch for BitNet ternary kernels.

#include "bitnet_kernel.h"

#if defined(__x86_64__) || defined(_M_X64)
#ifdef __GNUC__
#include <cpuid.h>
#endif
#endif

namespace nos {

static MatVecFn g_matvec_fn = nullptr;
static const char* g_backend_name = "none";

void bitnet_init() {
    g_matvec_fn = bitnet_matvec_scalar;
    g_backend_name = "scalar";

#if defined(__x86_64__) || defined(_M_X64)
#ifdef __GNUC__
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        bool has_avx512f = (ebx >> 16) & 1;
        bool has_avx512bw = (ebx >> 30) & 1;
        if (has_avx512f && has_avx512bw) {
#if NOS_HAS_AVX512_COMPILED
            g_matvec_fn = bitnet_matvec_avx512;
            g_backend_name = "avx512";
#endif
        }
    }
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#if NOS_HAS_NEON_COMPILED
    g_matvec_fn = bitnet_matvec_neon;
    g_backend_name = "neon";
#endif
#endif
}

void bitnet_matvec(const uint8_t* packed_weights, const float* input, float* output, int rows,
                   int cols, const uint16_t* scale_factors) {
    if (g_matvec_fn) {
        g_matvec_fn(packed_weights, input, output, rows, cols, scale_factors);
    }
}

const char* bitnet_get_backend_name() {
    return g_backend_name;
}

}  // namespace nos
