cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

message(STATUS "Building Editor Tests")

find_package(doctest REQUIRED)

# testCaptureConfig — tests AppConfig and CLI flag registration (no GPU needed)
# testGizmoScale — tests gizmo distance-based scaling (no GPU needed)
# testTransformDecompose — tests TRS matrix decompose/recompose roundtrip (no GPU needed)
set(TEST_NAMES testCaptureConfig testGizmoScale testTransformDecompose)
set(testCaptureConfig_source testCaptureConfig.cpp)
set(testGizmoScale_source testGizmoScale.cpp)
set(testTransformDecompose_source testTransformDecompose.cpp)

foreach(TEST_NAME IN LISTS TEST_NAMES)
    message(STATUS "Building test: ${TEST_NAME}")
    add_executable(${TEST_NAME} ${${TEST_NAME}_source})
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../generated
        ${CMAKE_SOURCE_DIR}
    )
    target_link_libraries(${TEST_NAME} PRIVATE
        doctest::doctest
        core
    )

    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
    )
    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests
        )
    endforeach()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# testDemoScenes — validates USDZ demo scene files open correctly (native only,
# reads files from the source tree).
if(NOT EMSCRIPTEN)
    set(_test testDemoScenes)
    message(STATUS "Building test: ${_test}")
    add_executable(${_test} testDemoScenes.cpp)
    target_compile_definitions(${_test} PRIVATE
        DEMO_SCENES_DIR=${CMAKE_SOURCE_DIR}/assets/scenes)
    target_link_libraries(${_test} PRIVATE doctest::doctest core)
    set_target_properties(${_test} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${_test} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
    endforeach()
    add_test(NAME ${_test} COMMAND ${_test})
endif()
