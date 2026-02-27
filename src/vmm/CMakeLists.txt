# -- nos_vmm: Virtual Memory Manager for expert caching --

add_library(nos_vmm STATIC
    slab_allocator.cpp
    clock_pro.cpp
    vmm.cpp
)

target_include_directories(nos_vmm PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(nos_vmm PUBLIC nos_io nos_format)

nos_set_warnings(nos_vmm)
