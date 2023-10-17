cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "core")

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
	tinyobjloader
	stb
	imgui
	expected
	nativefiledialog
	span
)

target_include_directories(${lib_name} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
)