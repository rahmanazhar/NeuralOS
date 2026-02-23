# ── nos_io: Platform-abstracted async I/O layer ──────────────────────────────

add_library(nos_io STATIC
    platform_io_stub.cpp
)

target_include_directories(nos_io
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

nos_set_warnings(nos_io)

# ── Linux: link with liburing ───────────────────────────────────────────────
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(nos_io PUBLIC ${URING_LIBRARIES})
    target_include_directories(nos_io PUBLIC ${URING_INCLUDE_DIRS})
endif()
