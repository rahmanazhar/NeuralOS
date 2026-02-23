# ── nos_format: Expert file format (.nxp) and CRC32 ──────────────────────────

add_library(nos_format STATIC
    expert_format_stub.cpp
)

target_include_directories(nos_format
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

nos_set_warnings(nos_format)

# ── Per-ISA CRC32 compile flags ─────────────────────────────────────────────
# When crc32.cpp is added by Plan 04:
#   if(NOS_HAS_SSE42)
#       set_source_files_properties(crc32.cpp PROPERTIES COMPILE_FLAGS "-msse4.2")
#   endif()
#   if(NOS_HAS_ARM_CRC)
#       set_source_files_properties(crc32.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crc")
#   endif()
