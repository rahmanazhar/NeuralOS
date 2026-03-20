# ── NeuralOS Shared Library (C API) ──────────────────────────────────────────

add_library(neuralos_shared SHARED libneuralos.cpp)

set_target_properties(neuralos_shared PROPERTIES
    OUTPUT_NAME "neuralos"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "libneuralos.h"
)

target_link_libraries(neuralos_shared
    PRIVATE
        nos_engine
        nos_tokenizer
        nos_vmm
        nos_kernel
        nos_io
        nos_format
)

target_include_directories(neuralos_shared
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

install(TARGETS neuralos_shared
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

nos_set_warnings(neuralos_shared)
