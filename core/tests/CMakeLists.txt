cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building Core Tests")

set(CMAKE_CXX_STANDARD 17)

# Find doctest (provided by conan)
find_package(doctest REQUIRED)

# Test executables
set(TEST_NAMES)

# Tests requiring native platform features (dynamic linking, browser WebGPU, boost)
if(NOT EMSCRIPTEN)
    list(APPEND TEST_NAMES testCommandLine)
    set(testCommandLine_source testCommandLine.cpp)
    set(testCommandLine_libs core)

    list(APPEND TEST_NAMES testWebGpu)
    set(testWebGpu_source testWebGpu.cpp)
    set(testWebGpu_libs core)
    set(testWebGpu_defines PTS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    list(APPEND TEST_NAMES testFrameGraph)
    set(testFrameGraph_source testFrameGraph.cpp)
    set(testFrameGraph_libs core)

    list(APPEND TEST_NAMES testAsyncStateMachine)
    set(testAsyncStateMachine_source testAsyncStateMachine.cpp)
    set(testAsyncStateMachine_libs core)

    list(APPEND TEST_NAMES testBufferReadback)
    set(testBufferReadback_source testBufferReadback.cpp)
    set(testBufferReadback_libs core)
endif()

# OpenUSD test - works on all platforms
list(APPEND TEST_NAMES testOpenUsd)
set(testOpenUsd_source testOpenUsd.cpp)
set(testOpenUsd_libs core)
set(testOpenUsd_includes ${CMAKE_CURRENT_SOURCE_DIR}/generated)

function(add_core_test target_name)
    message(STATUS "Building test: ${target_name}")
    add_executable(${target_name} ${${target_name}_source})
    target_include_directories(${target_name} PRIVATE
        ../include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${${target_name}_includes}
    )
    target_link_libraries(${target_name} PRIVATE
        doctest::doctest
        ${${target_name}_libs}
    )
    if(DEFINED ${target_name}_defines)
        target_compile_definitions(${target_name} PRIVATE ${${target_name}_defines})
    endif()

    if(EMSCRIPTEN)
        target_link_options(${target_name} PRIVATE
            "-sEXIT_RUNTIME=1"
            "-sPROXY_TO_PTHREAD=1"
        )
    endif()

    # Set output directory to bin/tests
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
    )

    # Ensure consistent output for multi-config generators
    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
        )
    endforeach()
endfunction()

foreach(TEST_NAME IN LISTS TEST_NAMES)
    add_core_test(${TEST_NAME})
endforeach()

# Enable CTest integration
include(CTest)
enable_testing()

# Register tests with CTest
foreach(TEST_NAME IN LISTS TEST_NAMES)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
