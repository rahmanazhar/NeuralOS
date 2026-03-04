# ── nos_io: Platform-abstracted async I/O layer ──────────────────────────────

set(NOS_IO_SOURCES
    pread_backend.cpp
    async_io_pool.cpp
)

# ── Linux: add io_uring backend ──────────────────────────────────────────────
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND NOS_IO_SOURCES io_uring_backend.cpp)
endif()

add_library(nos_io STATIC ${NOS_IO_SOURCES})

target_include_directories(nos_io
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

find_package(Threads REQUIRED)
target_link_libraries(nos_io PUBLIC Threads::Threads)

nos_set_warnings(nos_io)

# ── Linux: link with liburing, define NOS_HAS_IO_URING ──────────────────────
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(nos_io PUBLIC ${URING_LIBRARIES})
    target_include_directories(nos_io PUBLIC ${URING_INCLUDE_DIRS})
    target_compile_definitions(nos_io PUBLIC NOS_HAS_IO_URING)
endif()
