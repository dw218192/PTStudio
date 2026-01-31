cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building Core Tests")

set(CMAKE_CXX_STANDARD 17)

# Find doctest (provided by conan)
find_package(doctest REQUIRED)

# Test executables
set(TEST_NAMES
    testPluginSystem
    testOpenUsd
)

set(testPluginSystem_source testPluginSystem.cpp)
set(testPluginSystem_libs core)
set(testPluginSystem_includes ${CMAKE_SOURCE_DIR}/plugins/test_plugin/api)

set(testOpenUsd_source integration/testOpenUsd.cpp)
set(testOpenUsd_libs openusd::usd_usd)
set(testOpenUsd_includes ${PXR_INCLUDE_DIRS})

list(APPEND TEST_NAMES testWebGpu)
set(testWebGpu_source testWebGpu.cpp)
set(testWebGpu_libs core)
set(testWebGpu_defines PTS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

function(add_core_test target_name)
    message(STATUS "Building test: ${target_name}")
    add_executable(${target_name} ${${target_name}_source})
    target_include_directories(${target_name} PRIVATE
        ../include
        ${${target_name}_includes}
    )
    target_link_libraries(${target_name} PRIVATE
        doctest::doctest
        ${${target_name}_libs}
    )
    if(DEFINED ${target_name}_defines)
        target_compile_definitions(${target_name} PRIVATE ${${target_name}_defines})
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

# Static-plugin test executable (builds alongside the dynamic one).
set(STATIC_TEST_NAME testPluginSystemStatic)
message(STATUS "Building test: ${STATIC_TEST_NAME}")
add_executable(${STATIC_TEST_NAME} testPluginSystem.cpp)
target_include_directories(${STATIC_TEST_NAME} PRIVATE ../include)
target_link_libraries(${STATIC_TEST_NAME} PRIVATE doctest::doctest core test_plugin::api
    TestPluginStatic)
target_compile_definitions(${STATIC_TEST_NAME} PRIVATE PTS_STATIC_PLUGINS)

if (MSVC)
    target_link_libraries(${STATIC_TEST_NAME} PRIVATE TestPluginStatic)
    target_link_options(${STATIC_TEST_NAME} PRIVATE /WHOLEARCHIVE:TestPluginStatic)
elseif(APPLE)
    target_link_libraries(${STATIC_TEST_NAME} PRIVATE
        "-Wl,-force_load,$<TARGET_FILE:TestPluginStatic>")
else()
    target_link_libraries(${STATIC_TEST_NAME} PRIVATE
        "-Wl,--whole-archive" TestPluginStatic "-Wl,--no-whole-archive")
endif()

set_target_properties(${STATIC_TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
)

foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${config} config_upper)
    set_target_properties(${STATIC_TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
    )
endforeach()

# Enable CTest integration
include(CTest)
enable_testing()

# Register tests with CTest
foreach(TEST_NAME IN LISTS TEST_NAMES)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
add_test(NAME ${STATIC_TEST_NAME} COMMAND ${STATIC_TEST_NAME})