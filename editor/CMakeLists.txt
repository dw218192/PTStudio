cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "editor")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/imgui/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/imgui/*.*
)
#GLOB resource files
file(GLOB_RECURSE res_files 
	${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/icons/*.*
)

#Executable
add_executable(${exec_name} ${src_files})

#Embedded resources
cmrc_add_resource_library(${exec_name}_resources
	ALIAS ${exec_name}::rc
	${res_files}
)

#Libs linked to the executable
target_link_libraries(${exec_name}
PRIVATE
	$<BUILD_INTERFACE:developer_flags>
	core
	gl_utils
	glfw
	imguizmo
	ImGuiColorTextEdit
	Boost::stacktrace
	Boost::stacktrace_from_exception
	Boost::program_options
	#renderers
	vulkan_raytracer
	#resourcers
	${exec_name}::rc
)

target_include_directories(${exec_name}
PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

install(TARGETS ${exec_name}
	RUNTIME DESTINATION ${PROJECT_BINARY_DIR}
)