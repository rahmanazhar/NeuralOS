# -- nos_engine: Inference engine --

add_library(nos_engine STATIC
    rmsnorm.cpp
    rope.cpp
    attention.cpp
    router.cpp
    inference_engine.cpp
    perplexity.cpp
    sampling.cpp
    thread_pool.cpp
    shift_detector.cpp
)

target_include_directories(nos_engine PUBLIC ${CMAKE_SOURCE_DIR}/src)
find_package(Threads REQUIRED)
target_link_libraries(nos_engine PUBLIC nos_vmm nos_kernel nos_converter Threads::Threads)
nos_set_warnings(nos_engine)
