# ── nos_kernel: BitNet b1.58 ternary compute kernels ─────────────────────────

add_library(nos_kernel STATIC
    bitnet_dispatch.cpp
)

target_include_directories(nos_kernel
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

nos_set_warnings(nos_kernel)

# Link with nos_format for packing utilities
target_link_libraries(nos_kernel
    PUBLIC nos_format
)

# ── Per-ISA compile flags ───────────────────────────────────────────────────
# Files matching *_avx512.cpp get AVX-512 flags (added by Plan 02)
# Files matching *_neon.cpp get no extra flags on AArch64 (NEON is baseline)
#
# Example for Plan 02:
#   set_source_files_properties(bitnet_avx512.cpp PROPERTIES
#       COMPILE_FLAGS "-mavx512f -mavx512bw"
#   )
