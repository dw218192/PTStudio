cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "editor")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/imgui/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/imgui/*.*
)

#Executable
add_executable(${exec_name} ${src_files})

#Libs linked to the executable
target_link_libraries(${exec_name}   
	$<BUILD_INTERFACE:developer_flags>
	core
	gl_wrapper
	glfw
	imguizmo
	ImGuiColorTextEdit
)

install(TARGETS ${exec_name}
	RUNTIME DESTINATION ${PROJECT_BINARY_DIR}
)