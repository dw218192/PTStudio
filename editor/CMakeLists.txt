cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "editor")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/imgui/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/imgui/*.*
)
list(FILTER src_files EXCLUDE REGEX "editorRenderer")

#Executable
add_executable(${exec_name} ${src_files})

#Libs linked to the executable
target_link_libraries(${exec_name}
PRIVATE
	$<BUILD_INTERFACE:developer_flags>
	core
	imguizmo::imguizmo
	stb::stb
)
if (MSVC)
	target_compile_options(${exec_name} PRIVATE /bigobj)
endif()

target_include_directories(${exec_name}
PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/generated
	${CMAKE_SOURCE_DIR}  # For generated headers (core/generated, editor/generated)
)

install(TARGETS ${exec_name}
	RUNTIME DESTINATION ${PROJECT_BINARY_DIR}
)