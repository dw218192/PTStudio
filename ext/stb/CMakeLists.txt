cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "pts_stb")

add_library(${lib_name} STATIC stb_impl.cpp)

# Use headers from the Conan-provided stb package
target_link_libraries(${lib_name}
    PUBLIC
        stb::stb
)

target_include_directories(${lib_name}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


