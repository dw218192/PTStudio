cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "core")

#GLOB source files 
file(GLOB_RECURSE src_files
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/imgui/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/imgui/*.*
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
	fmt-header-only
	nativefiledialog
	span
	nlohmann_json
)

set(PT_CORE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/include PARENT_SCOPE)

target_include_directories(${lib_name} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

# Build unit tests if enabled
if (CORE_BUILD_TESTS)
	add_subdirectory(tests)
endif()