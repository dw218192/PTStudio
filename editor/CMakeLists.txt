#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "editor")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
)

#Executable
add_executable(${exec_name} ${src_files})

#Libs linked to the executable
target_link_libraries(${exec_name}   
	$<BUILD_INTERFACE:developer_flags>
	glfw
	libglew_static
	glm
	tinyobjloader
	stb
	imgui
	expected
)