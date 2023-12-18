cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building Core Tests")

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(${EXT_DIR}/Catch2 ${CMAKE_BINARY_DIR}/Catch2)
list(APPEND TEST_NAMES
    test_reflection    
    test_type_traits
)

foreach(TEST_NAME IN LISTS TEST_NAMES)
    message(STATUS "Building test: ${TEST_NAME}")
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ../src/include ${EXT_DIR}/Catch2/single_include)
    target_link_libraries(${TEST_NAME} PRIVATE Catch2::Catch2)
endforeach()