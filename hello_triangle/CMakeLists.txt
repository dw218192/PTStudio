cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(hello_triangle)

add_executable(hello_triangle
    src/main.cpp
)

target_link_libraries(hello_triangle
    PRIVATE
        core
)

# Set output directory
set_target_properties(hello_triangle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
