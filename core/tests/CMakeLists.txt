cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building Core Tests")

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(${EXT_DIR}/Catch2 ${CMAKE_BINARY_DIR}/Catch2)
add_executable(test_reflection
    test_reflection.cpp
    ../src/include/reflection.h)
target_include_directories(test_reflection PRIVATE ../src/include ${EXT_DIR}/Catch2/single_include)