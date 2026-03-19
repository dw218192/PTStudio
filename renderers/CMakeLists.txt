cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# add_renderer(target_name SOURCES src1 src2 ...)
# Creates an OBJECT library with standard include dirs and core linkage.
macro(add_renderer TARGET_NAME)
    cmake_parse_arguments(REND "" "" "SOURCES" ${ARGN})
    add_library(${TARGET_NAME} OBJECT ${REND_SOURCES})
    target_include_directories(${TARGET_NAME}
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/generated
        PRIVATE ${CMAKE_SOURCE_DIR}
    )
    target_link_libraries(${TARGET_NAME}
        PUBLIC  core
        PRIVATE $<BUILD_INTERFACE:developer_flags>
    )
    if (MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /bigobj)
    endif()
endmacro()

# Future renderer subdirectories go here:
# add_subdirectory(deferred)
