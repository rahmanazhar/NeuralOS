# ── nos_kernel: BitNet b1.58 ternary compute kernels ─────────────────────────

set(KERNEL_SOURCES
    bitnet_dispatch.cpp
    bitnet_scalar.cpp
    packing.cpp
)

# ── Per-ISA source files with isolated compile flags ────────────────────────

if(NOS_HAS_AVX512)
    list(APPEND KERNEL_SOURCES bitnet_avx512.cpp)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    list(APPEND KERNEL_SOURCES bitnet_neon.cpp)
endif()

add_library(nos_kernel STATIC ${KERNEL_SOURCES})

target_include_directories(nos_kernel
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

nos_set_warnings(nos_kernel)

# Link with nos_format for packing utilities
target_link_libraries(nos_kernel
    PUBLIC nos_format
)

# ── Per-ISA compile flags (isolated per translation unit) ───────────────────

if(NOS_HAS_AVX512)
    set_source_files_properties(bitnet_avx512.cpp PROPERTIES
        COMPILE_FLAGS "-mavx512f -mavx512bw"
    )
    target_compile_definitions(nos_kernel PRIVATE NOS_HAS_AVX512_COMPILED=1)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    target_compile_definitions(nos_kernel PRIVATE NOS_HAS_NEON_COMPILED=1)
endif()
