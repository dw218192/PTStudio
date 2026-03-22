cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building Core Tests")

set(CMAKE_CXX_STANDARD 17)

# Find doctest (provided by conan)
find_package(doctest REQUIRED)

# Test executables
set(TEST_NAMES)

# Non-GPU tests — run on all platforms including Emscripten (via node.js)
list(APPEND TEST_NAMES testCommandLine)
set(testCommandLine_source testCommandLine.cpp)
set(testCommandLine_libs core)

list(APPEND TEST_NAMES testAsyncStateMachine)
set(testAsyncStateMachine_source testAsyncStateMachine.cpp)
set(testAsyncStateMachine_libs core)

# GPU tests — require native WebGPU (Dawn); no browser/node.js GPU access
if(NOT EMSCRIPTEN)
    list(APPEND TEST_NAMES testWebGpu)
    set(testWebGpu_source testWebGpu.cpp)
    set(testWebGpu_libs core)
    set(testWebGpu_defines PTS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    list(APPEND TEST_NAMES testFrameGraph)
    set(testFrameGraph_source testFrameGraph.cpp)
    set(testFrameGraph_libs core)

    list(APPEND TEST_NAMES testBufferReadback)
    set(testBufferReadback_source testBufferReadback.cpp)
    set(testBufferReadback_libs core)

    list(APPEND TEST_NAMES testMaterialBuffer)
    set(testMaterialBuffer_source testMaterialBuffer.cpp)
    set(testMaterialBuffer_libs core)

    list(APPEND TEST_NAMES testPipelineBuilder)
    set(testPipelineBuilder_source testPipelineBuilder.cpp)
    set(testPipelineBuilder_libs core)

    # Geometric adapter tests - sync() requires GPU for mesh upload
    list(APPEND TEST_NAMES testGeometricAdapters)
    set(testGeometricAdapters_source testGeometricAdapters.cpp)
    set(testGeometricAdapters_libs core)
    set(testGeometricAdapters_includes ${CMAKE_CURRENT_SOURCE_DIR}/generated)
endif()

# PropertyDescriptor / get_properties - no GPU needed, just USD stages
list(APPEND TEST_NAMES testPropertyDescriptor)
set(testPropertyDescriptor_source testPropertyDescriptor.cpp)
set(testPropertyDescriptor_libs core)

# OpenUSD test - works on all platforms
list(APPEND TEST_NAMES testOpenUsd)
set(testOpenUsd_source testOpenUsd.cpp)
set(testOpenUsd_libs core)
set(testOpenUsd_includes ${CMAKE_CURRENT_SOURCE_DIR}/generated)

# Wireframe index expansion - works on all platforms (no GPU needed)
list(APPEND TEST_NAMES testWireframeIndices)
set(testWireframeIndices_source testWireframeIndices.cpp)
set(testWireframeIndices_libs core)

# Mesh cache service on IScenePass - works on all platforms (no GPU needed)
list(APPEND TEST_NAMES testMeshCache)
set(testMeshCache_source testMeshCache.cpp)
set(testMeshCache_libs core)

# RenderWorld slot map - works on all platforms (no GPU needed)
list(APPEND TEST_NAMES testRenderWorldSlotMap)
set(testRenderWorldSlotMap_source testRenderWorldSlotMap.cpp)
set(testRenderWorldSlotMap_libs core)

# RendererRegistry - works on all platforms (no GPU needed)
list(APPEND TEST_NAMES testRendererRegistry)
set(testRendererRegistry_source testRendererRegistry.cpp)
set(testRendererRegistry_libs core)

# ShaderLoader - works on all platforms (no GPU needed)
list(APPEND TEST_NAMES testShaderLoader)
set(testShaderLoader_source testShaderLoader.cpp)
set(testShaderLoader_libs core)
set(testShaderLoader_includes ${CMAKE_CURRENT_SOURCE_DIR}/generated)

# BackgroundTask - works on all platforms (no GPU needed)
list(APPEND TEST_NAMES testBackgroundTask)
set(testBackgroundTask_source testBackgroundTask.cpp)
set(testBackgroundTask_libs core)

# LoadingOverlay - no GPU needed, tests non-draw logic
list(APPEND TEST_NAMES testLoadingOverlay)
set(testLoadingOverlay_source testLoadingOverlay.cpp)
set(testLoadingOverlay_libs core)


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
