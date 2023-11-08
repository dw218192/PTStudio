cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "gl_wrapper")

#GLOB source files 
file(GLOB_RECURSE src_files
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
)

#Executable
add_library(${lib_name} STATIC ${src_files})

#Libs linked to the executable
target_link_libraries(${lib_name}   
	$<BUILD_INTERFACE:developer_flags>
    glm
    libglew_static
    expected
    span
)
target_include_directories(${lib_name} PUBLIC
    ${PT_CORE_INCLUDE_DIR}
    ${STB_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
)