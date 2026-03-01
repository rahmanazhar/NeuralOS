# ── NeuralOS CLI ─────────────────────────────────────────────────────────────

add_executable(neuralos main.cpp)

target_link_libraries(neuralos
    PRIVATE
        nos_engine
        nos_converter
        nos_tokenizer
        nos_vmm
        nos_kernel
        nos_io
        nos_format
)

target_include_directories(neuralos
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
