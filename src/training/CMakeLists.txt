# -- nos_training: Edge training infrastructure --

add_library(nos_training STATIC
    data_loader.cpp
    badam.cpp
    galore.cpp
)

target_include_directories(nos_training PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nos_training PUBLIC nos_engine nos_tokenizer nlohmann_json::nlohmann_json)
nos_set_warnings(nos_training)
