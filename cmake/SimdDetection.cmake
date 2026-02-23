# ── SIMD Capability Detection ────────────────────────────────────────────────
#
# Detects SIMD capabilities by test-compiling actual intrinsic code (GROMACS
# pattern). Sets cache variables:
#   NOS_HAS_AVX512       - AVX-512F + AVX-512BW
#   NOS_HAS_AVX512_VNNI  - AVX-512 VNNI
#   NOS_HAS_SSE42        - SSE4.2 (for CRC32C)
#   NOS_HAS_NEON         - ARM NEON (always ON on AArch64)
#   NOS_HAS_ARM_CRC      - ARM CRC extension

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# ── x86-64 SIMD Detection ──────────────────────────────────────────────────

# AVX-512F + AVX-512BW
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw")
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m512i a = _mm512_setzero_si512();
    __m512i b = _mm512_setzero_si512();
    __m512i c = _mm512_add_epi16(a, b);
    (void)c;
    return 0;
}
" NOS_HAS_AVX512)
unset(CMAKE_REQUIRED_FLAGS)

# AVX-512 VNNI
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vnni")
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m512i a = _mm512_setzero_si512();
    __m512i b = _mm512_setzero_si512();
    __m512i c = _mm512_setzero_si512();
    __m512i d = _mm512_dpbusd_epi32(a, b, c);
    (void)d;
    return 0;
}
" NOS_HAS_AVX512_VNNI)
unset(CMAKE_REQUIRED_FLAGS)

# SSE4.2 (for CRC32C hardware acceleration)
set(CMAKE_REQUIRED_FLAGS "-msse4.2")
check_cxx_source_compiles("
#include <nmmintrin.h>
#include <cstdint>
int main() {
    uint64_t val = 42;
    uint32_t crc = _mm_crc32_u64(0, val);
    (void)crc;
    return 0;
}
" NOS_HAS_SSE42)
unset(CMAKE_REQUIRED_FLAGS)

# ── AArch64 SIMD Detection ─────────────────────────────────────────────────

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # NEON is always available on AArch64
    set(NOS_HAS_NEON ON CACHE BOOL "ARM NEON is available (baseline on AArch64)")

    # ARM CRC extension
    set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+crc")
    check_cxx_source_compiles("
#include <arm_acle.h>
#include <cstdint>
int main() {
    uint64_t val = 42;
    uint32_t crc = __crc32cd(0, val);
    (void)crc;
    return 0;
}
" NOS_HAS_ARM_CRC)
    unset(CMAKE_REQUIRED_FLAGS)
else()
    set(NOS_HAS_NEON OFF CACHE BOOL "ARM NEON is not available (not AArch64)")
    set(NOS_HAS_ARM_CRC OFF CACHE BOOL "ARM CRC is not available (not AArch64)")
endif()

# ── Summary ─────────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "=== NeuralOS SIMD Detection ===")
message(STATUS "  Processor:      ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  AVX-512:        ${NOS_HAS_AVX512}")
message(STATUS "  AVX-512 VNNI:   ${NOS_HAS_AVX512_VNNI}")
message(STATUS "  SSE4.2:         ${NOS_HAS_SSE42}")
message(STATUS "  NEON:           ${NOS_HAS_NEON}")
message(STATUS "  ARM CRC:        ${NOS_HAS_ARM_CRC}")
message(STATUS "===============================")
message(STATUS "")
