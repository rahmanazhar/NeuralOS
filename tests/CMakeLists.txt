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

# ── Quantizer Tests ──────────────────────────────────────────────────
add_executable(test_quantizer test_quantizer.cpp)

target_link_libraries(test_quantizer
    PRIVATE
        Catch2::Catch2WithMain
        nos_converter
)

catch_discover_tests(test_quantizer)

# ── Conversion Pipeline Tests ────────────────────────────────────────
add_executable(test_conversion_pipeline test_conversion_pipeline.cpp)

target_link_libraries(test_conversion_pipeline
    PRIVATE
        Catch2::Catch2WithMain
        nos_converter
)

catch_discover_tests(test_conversion_pipeline)

# ── RMSNorm Tests ───────────────────────────────────────────────────
add_executable(test_rmsnorm test_rmsnorm.cpp)

target_link_libraries(test_rmsnorm
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_rmsnorm)

# ── RoPE Tests ──────────────────────────────────────────────────────
add_executable(test_rope test_rope.cpp)

target_link_libraries(test_rope
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_rope)

# ── Attention Tests ─────────────────────────────────────────────────
add_executable(test_attention test_attention.cpp)

target_link_libraries(test_attention
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_attention)

# ── Router Tests ────────────────────────────────────────────────────
add_executable(test_router test_router.cpp)

target_link_libraries(test_router
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_router)

# ── Inference Engine Tests ──────────────────────────────────────────
add_executable(test_inference_engine test_inference_engine.cpp)

target_link_libraries(test_inference_engine
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_inference_engine)

# ── Sampling Tests ─────────────────────────────────────────────────
add_executable(test_sampling test_sampling.cpp)

target_link_libraries(test_sampling
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_sampling)

# ── End-to-End Synthetic Tests ─────────────────────────────────────
add_executable(test_e2e_synthetic test_e2e_synthetic.cpp)

target_link_libraries(test_e2e_synthetic
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_e2e_synthetic)

# ── Thread Pool Tests ─────────────────────────────────────────
add_executable(test_thread_pool test_thread_pool.cpp)

target_link_libraries(test_thread_pool
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_thread_pool)

# ── Async VMM Tests ───────────────────────────────────────────
add_executable(test_async_vmm test_async_vmm.cpp)

target_link_libraries(test_async_vmm
    PRIVATE
        Catch2::Catch2WithMain
        nos_vmm
)

catch_discover_tests(test_async_vmm)

# ── Shift Detector Tests ────────────────────────────────────────
add_executable(test_shift_detector test_shift_detector.cpp)

target_link_libraries(test_shift_detector
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_shift_detector)

# ── Sticky Router Tests ─────────────────────────────────────────
add_executable(test_sticky_router test_sticky_router.cpp)

target_link_libraries(test_sticky_router
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_sticky_router)

# ── Metrics Tests ──────────────────────────────────────────────
add_executable(test_metrics test_metrics.cpp)

target_link_libraries(test_metrics
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_metrics)

# ── Benchmark Output Tests ─────────────────────────────────────
add_executable(test_benchmark_output test_benchmark_output.cpp)

target_link_libraries(test_benchmark_output
    PRIVATE
        Catch2::Catch2WithMain
        nos_engine
)

catch_discover_tests(test_benchmark_output)

# -- Oracle LSTM Tests --
add_executable(test_oracle_lstm test_oracle_lstm.cpp)
target_link_libraries(test_oracle_lstm PRIVATE Catch2::Catch2WithMain nos_engine)
catch_discover_tests(test_oracle_lstm)

# -- Prefetch Predictor Tests --
add_executable(test_prefetch_predictor test_prefetch_predictor.cpp)
target_link_libraries(test_prefetch_predictor PRIVATE Catch2::Catch2WithMain nos_engine)
catch_discover_tests(test_prefetch_predictor)

# -- Oracle Prefetcher Tests --
add_executable(test_oracle_prefetcher test_oracle_prefetcher.cpp)
target_link_libraries(test_oracle_prefetcher PRIVATE Catch2::Catch2WithMain nos_engine)
catch_discover_tests(test_oracle_prefetcher)

# ── libneuralos C API Tests ────────────────────────────────────────────
add_executable(test_libneuralos test_libneuralos.cpp)
target_link_libraries(test_libneuralos PRIVATE Catch2::Catch2WithMain neuralos_shared)
target_include_directories(test_libneuralos PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_libneuralos)

# ── Chat Template Tests ───────────────────────────────────────────────
add_executable(test_chat_template test_chat_template.cpp)
target_link_libraries(test_chat_template PRIVATE Catch2::Catch2WithMain nos_server)
target_include_directories(test_chat_template PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_chat_template)

# ── HTTP Server Tests ─────────────────────────────────────────────────
add_executable(test_http_server test_http_server.cpp)
target_link_libraries(test_http_server PRIVATE Catch2::Catch2WithMain nos_server)
target_include_directories(test_http_server PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_http_server)

# ── DataLoader Tests ─────────────────────────────────────────────────
add_executable(test_data_loader test_data_loader.cpp)
target_link_libraries(test_data_loader PRIVATE Catch2::Catch2WithMain nos_training)
target_include_directories(test_data_loader PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_data_loader)

# ── BAdam Tests ──────────────────────────────────────────────────────
add_executable(test_badam test_badam.cpp)
target_link_libraries(test_badam PRIVATE Catch2::Catch2WithMain nos_training)
target_include_directories(test_badam PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_badam)

# ── GaLore Tests ─────────────────────────────────────────────────────
add_executable(test_galore test_galore.cpp)
target_link_libraries(test_galore PRIVATE Catch2::Catch2WithMain nos_training)
target_include_directories(test_galore PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_galore)

# ── LoRA Tests ───────────────────────────────────────────────────────
add_executable(test_lora test_lora.cpp)
target_link_libraries(test_lora PRIVATE Catch2::Catch2WithMain nos_training)
target_include_directories(test_lora PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_lora)

# ── Sparkline Tests ──────────────────────────────────────────────────
add_executable(test_sparkline test_sparkline.cpp)
target_link_libraries(test_sparkline PRIVATE Catch2::Catch2WithMain nos_dashboard)
target_include_directories(test_sparkline PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_sparkline)

# ── SharedMetrics Tests ──────────────────────────────────────────────
add_executable(test_shared_metrics test_shared_metrics.cpp)
target_link_libraries(test_shared_metrics PRIVATE Catch2::Catch2WithMain nos_server)
target_include_directories(test_shared_metrics PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_shared_metrics)

# ── RequestScheduler Tests ───────────────────────────────────────────
add_executable(test_request_scheduler test_request_scheduler.cpp)
target_link_libraries(test_request_scheduler PRIVATE Catch2::Catch2WithMain nos_server)
target_include_directories(test_request_scheduler PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_request_scheduler)

# ── Bench Command Tests ─────────────────────────────────────────────
add_executable(test_bench_command test_bench_command.cpp)
target_link_libraries(test_bench_command PRIVATE Catch2::Catch2WithMain nos_engine)
target_include_directories(test_bench_command PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(test_bench_command)
