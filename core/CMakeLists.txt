cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "core")

#GLOB source files
set(core_src_globs
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/imgui/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/imgui/*.*
)
file(GLOB_RECURSE core_src_files ${core_src_globs})
file(GLOB rendering_root_files ${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/*.*)
list(FILTER core_src_files EXCLUDE REGEX "/rendering/")
list(APPEND core_src_files ${rendering_root_files})

set(core_rendering_sources)

if(PTS_WINDOWING STREQUAL "glfw")
	file(GLOB_RECURSE core_windowing_sources
		${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/glfw/*.*
	)
	list(APPEND core_rendering_sources ${core_windowing_sources})
elseif(PTS_WINDOWING STREQUAL "null")
	file(GLOB_RECURSE core_windowing_sources
		${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/null/*.*
	)
	list(APPEND core_rendering_sources ${core_windowing_sources})
endif()

set(src_files ${core_src_files} ${core_rendering_sources})
list(REMOVE_DUPLICATES src_files)

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
	spdlog::spdlog
	nativefiledialog
	tcb-span::tcb-span
	nlohmann_json::nlohmann_json
	Boost::filesystem
	Boost::container
)

if(PTS_WINDOWING STREQUAL "glfw")
	target_link_libraries(${lib_name} PUBLIC glfw)
endif()

if (MSVC)
	target_compile_options(${lib_name} PRIVATE /bigobj)
endif()

set(PT_CORE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/include PARENT_SCOPE)

target_include_directories(${lib_name}
PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/api
PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/include/core
)

# ============================================================================
# ABI-Stable API Interface Target
# ============================================================================
# This interface library provides ABI-stable public headers for use by dyanmic libraries

add_library(core_api INTERFACE)
add_library(core::api ALIAS core_api)

target_include_directories(core_api
INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/api
)

target_link_libraries(core_api
INTERFACE
	fmt::fmt
)

# Build unit tests if enabled
if (CORE_BUILD_TESTS)
	add_subdirectory(tests)
endif()