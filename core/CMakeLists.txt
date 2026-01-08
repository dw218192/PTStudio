cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "core")

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
)

#Executable
add_library(${lib_name} STATIC ${src_files})

#Resources
cmrc_add_resource_library(${lib_name}_resources
	ALIAS ${lib_name}::rc
	${res_files}
)

#Libs linked to the executable
target_link_libraries(${lib_name}
PRIVATE
	$<BUILD_INTERFACE:developer_flags>
PUBLIC
	${lib_name}::rc
	glm::glm
	tinyobjloader
	pts_stb
	imgui
	tl::expected
	fmt::fmt
	nativefiledialog
	tcb-span::tcb-span
	nlohmann_json::nlohmann_json
)

set(PT_CORE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/include PARENT_SCOPE)

target_include_directories(${lib_name} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# Build unit tests if enabled
# if (CORE_BUILD_TESTS)
# 	add_subdirectory(tests)
# endif()