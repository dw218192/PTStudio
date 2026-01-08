cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "gl_wrapper")

# GLOB source files 
file(GLOB_RECURSE src_files
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
)

# Library
add_library(${lib_name} STATIC ${src_files})

target_include_directories(${lib_name}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/src/include
)

target_link_libraries(${lib_name} PRIVATE
  $<BUILD_INTERFACE:developer_flags>
)
target_link_libraries(${lib_name} PUBLIC
    glm::glm
    GLEW::glew_s
    tl::expected
    fmt::fmt
    tcb-span::tcb-span
    pts_stb
)

