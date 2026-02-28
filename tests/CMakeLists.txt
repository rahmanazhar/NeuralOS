# ── NeuralOS Tests ───────────────────────────────────────────────────────────

include(CTest)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

# ── Smoke Test ──────────────────────────────────────────────────────────────
add_executable(test_smoke test_smoke.cpp)

target_link_libraries(test_smoke
    PRIVATE
        Catch2::Catch2WithMain
        nos_format
)

target_include_directories(test_smoke
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/format
)

catch_discover_tests(test_smoke)

# ── Platform I/O Tests ──────────────────────────────────────────────────────
add_executable(test_platform_io test_platform_io.cpp)

target_link_libraries(test_platform_io
    PRIVATE
        Catch2::Catch2WithMain
        nos_io
)

target_include_directories(test_platform_io
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/io
)

catch_discover_tests(test_platform_io)

# ── Packing Tests ──────────────────────────────────────────────────────────
add_executable(test_packing test_packing.cpp)

target_link_libraries(test_packing
    PRIVATE
        Catch2::Catch2WithMain
        nos_kernel
)

target_include_directories(test_packing
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernel
)

catch_discover_tests(test_packing)

# ── BitNet Kernel Tests ────────────────────────────────────────────────────
add_executable(test_bitnet_kernel test_bitnet_kernel.cpp)

target_link_libraries(test_bitnet_kernel
    PRIVATE
        Catch2::Catch2WithMain
        nos_kernel
)

target_include_directories(test_bitnet_kernel
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernel
)

catch_discover_tests(test_bitnet_kernel)

# ── Packing Benchmark ─────────────────────────────────────────────────────
add_executable(bench_packing bench_packing.cpp)

target_link_libraries(bench_packing
    PRIVATE
        Catch2::Catch2WithMain
        nos_kernel
)

target_include_directories(bench_packing
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernel
)

# Benchmarks are tagged with [.benchmark] so they don't run by default.
# Run manually: ./build/tests/bench_packing --benchmark-no-analysis
catch_discover_tests(bench_packing)

# ── BitNet Kernel Benchmark ──────────────────────────────────────────────────
add_executable(bench_bitnet_kernel bench_bitnet_kernel.cpp)

target_link_libraries(bench_bitnet_kernel
    PRIVATE
        Catch2::Catch2WithMain
        nos_kernel
)

target_include_directories(bench_bitnet_kernel
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernel
)

catch_discover_tests(bench_bitnet_kernel)

# ── CRC32 Tests ─────────────────────────────────────────────────────────────
add_executable(test_crc32 test_crc32.cpp)

target_link_libraries(test_crc32
    PRIVATE
        Catch2::Catch2WithMain
        nos_format
)

target_include_directories(test_crc32
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/format
)

catch_discover_tests(test_crc32)

# ── Expert Format Tests ─────────────────────────────────────────────────────
add_executable(test_expert_format test_expert_format.cpp)

target_link_libraries(test_expert_format
    PRIVATE
        Catch2::Catch2WithMain
        nos_format
)

target_include_directories(test_expert_format
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/format
)

catch_discover_tests(test_expert_format)

# ── Tokenizer Tests ─────────────────────────────────────────────────────────
add_executable(test_tokenizer test_tokenizer.cpp)

target_link_libraries(test_tokenizer
    PRIVATE
        Catch2::Catch2WithMain
        nos_tokenizer
)

target_include_directories(test_tokenizer
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/tokenizer
)

catch_discover_tests(test_tokenizer)

# ── Slab Allocator Tests ─────────────────────────────────────────────────
add_executable(test_slab_allocator test_slab_allocator.cpp)

target_link_libraries(test_slab_allocator
    PRIVATE
        Catch2::Catch2WithMain
        nos_vmm
)

catch_discover_tests(test_slab_allocator)

# ── CLOCK-Pro Tests ──────────────────────────────────────────────────────
add_executable(test_clock_pro test_clock_pro.cpp)

target_link_libraries(test_clock_pro
    PRIVATE
        Catch2::Catch2WithMain
        nos_vmm
)

catch_discover_tests(test_clock_pro)

# ── VMM Tests ────────────────────────────────────────────────────────────
add_executable(test_vmm test_vmm.cpp)

target_link_libraries(test_vmm
    PRIVATE
        Catch2::Catch2WithMain
        nos_vmm
)

catch_discover_tests(test_vmm)

# ── Memory Budget Tests ────────────────────────────────────────────────
add_executable(test_memory_budget test_memory_budget.cpp)

target_link_libraries(test_memory_budget
    PRIVATE
        Catch2::Catch2WithMain
        nos_vmm
)

catch_discover_tests(test_memory_budget)

# ── VMM Budget Integration Tests ──────────────────────────────────────
add_executable(test_vmm_budget test_vmm_budget.cpp)

target_link_libraries(test_vmm_budget
    PRIVATE
        Catch2::Catch2WithMain
        nos_vmm
)

catch_discover_tests(test_vmm_budget)

# ── SafeTensors Reader Tests ─────────────────────────────────────────
add_executable(test_safetensors_reader test_safetensors_reader.cpp)

target_link_libraries(test_safetensors_reader
    PRIVATE
        Catch2::Catch2WithMain
        nos_converter
)

catch_discover_tests(test_safetensors_reader)

# ── GGUF Reader Tests ────────────────────────────────────────────────
add_executable(test_gguf_reader test_gguf_reader.cpp)

target_link_libraries(test_gguf_reader
    PRIVATE
        Catch2::Catch2WithMain
        nos_converter
)

catch_discover_tests(test_gguf_reader)

# ── Model Config Tests ───────────────────────────────────────────────
add_executable(test_model_config test_model_config.cpp)

target_link_libraries(test_model_config
    PRIVATE
        Catch2::Catch2WithMain
        nos_converter
)

catch_discover_tests(test_model_config)
