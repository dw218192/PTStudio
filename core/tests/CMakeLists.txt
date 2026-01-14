cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building Core Tests")

set(CMAKE_CXX_STANDARD 17)

# Find doctest (provided by conan)
find_package(doctest REQUIRED)

# Test executables
set(TEST_NAMES
    testReflection    
    testTypeTraits
    testPluginSystem
)

foreach(TEST_NAME IN LISTS TEST_NAMES)
    message(STATUS "Building test: ${TEST_NAME}")
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ../include)
    target_link_libraries(${TEST_NAME} PRIVATE doctest::doctest)
    
    # testPluginSystem needs core library and plugin interface headers
    if(TEST_NAME STREQUAL "testPluginSystem")
        target_link_libraries(${TEST_NAME} PRIVATE core)
        target_include_directories(${TEST_NAME} PRIVATE 
            ${CMAKE_SOURCE_DIR}/plugins/test_plugin/api
        )
    endif()
    
    # Set output directory to bin/tests
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
    )
    
    # Ensure consistent output for multi-config generators
    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
        )
    endforeach()
endforeach()

# Enable CTest integration
include(CTest)
enable_testing()

# Register tests with CTest
foreach(TEST_NAME IN LISTS TEST_NAMES)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()