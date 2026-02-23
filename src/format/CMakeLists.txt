# -- nos_format: Expert file format (.nxp) and CRC32 --

add_library(nos_format STATIC
    crc32.cpp
    expert_writer.cpp
    expert_reader.cpp
)

target_include_directories(nos_format
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

nos_set_warnings(nos_format)

# -- Per-ISA CRC32 compile flags --
if(NOS_HAS_SSE42)
    set_source_files_properties(crc32.cpp PROPERTIES COMPILE_FLAGS "-msse4.2")
endif()

if(NOS_HAS_ARM_CRC)
    set_source_files_properties(crc32.cpp PROPERTIES COMPILE_FLAGS "-march=armv8-a+crc")
endif()
