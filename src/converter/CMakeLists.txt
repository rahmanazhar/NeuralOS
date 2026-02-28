# -- nos_converter: Model conversion pipeline --

add_library(nos_converter STATIC
    safetensors_reader.cpp
    gguf_reader.cpp
    model_reader.cpp
)

target_include_directories(nos_converter PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nos_converter PUBLIC nlohmann_json::nlohmann_json nos_format nos_kernel)
nos_set_warnings(nos_converter)
