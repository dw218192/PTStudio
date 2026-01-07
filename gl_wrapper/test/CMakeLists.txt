cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building GL Wrapper Tests")

set(CMAKE_CXX_STANDARD 17)

# Find doctest
find_package(doctest REQUIRED)

# Test executable
add_executable(gl_wrapper_test test_gl_wrapper.cpp)

target_include_directories(gl_wrapper_test PRIVATE 
    ../src/include
)

target_link_libraries(gl_wrapper_test PRIVATE
    gl_wrapper
    doctest::doctest
)

# Set output directory
set_target_properties(gl_wrapper_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

