#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

#Project name 
set(project_name "PTStudio")
project(${project_name} LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

#default build type is Debug
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug)
endif ()

# options
# Use if(NOT DEFINED) to allow Conan cache variables to take precedence
if(NOT DEFINED CORE_BUILD_TESTS)
    option(CORE_BUILD_TESTS "Build core unit tests" ON)
endif()

# Direct all output to /bin directory
# Since we organize builds by configuration (build/Debug, build/Release) -- see build.ps1,
# we don't want CMake to append the config name again
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
# Prevent multi-config generators from adding config suffix
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Auto-detect GPU architecture (only if building path tracer)

# C++ compiler flags
set(cxx_flags
	$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /std:c++17 /MP>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/MDd /Zi /Ob0 /Od /RTC1>
	# Add MSVC-specific compiler flags here
	
	$<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -O3 -std=c++17 -fmax-errors=1>
	$<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-g -O0>
	# Add GCC compiler flags here
	
	$<$<CXX_COMPILER_ID:Clang>:-Wall -m64 -O3 -std=c++17 -ferror-limit=1>
	$<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:-g -O0>
	# Add Clang compiler flags here
)

set(MSVC_XCOMPILER_FLAGS
	"/std:c++17"
	$<$<CONFIG:Debug>:/MDd /Zi /Ob0 /Od /RTC1>
)

add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
)

target_compile_features(developer_flags INTERFACE cxx_std_17)

set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext")

find_package(glfw3 REQUIRED)
find_package(glew REQUIRED)
find_package(glm REQUIRED)
find_package(tinyobjloader REQUIRED)
find_package(stb REQUIRED)
find_package(tcb-span REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(tl-expected REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

# Vulkan support
find_package(VulkanHeaders REQUIRED)
find_package(Vulkan REQUIRED)
find_package(shaderc REQUIRED)

# external dependencies that are built from source
add_subdirectory(${EXT_DIR})

# add resource compiler
include(${EXT_DIR}/cmrc/CMakeRC.cmake)

# add subprojects
add_subdirectory(core)

# renderers
add_subdirectory(gl_utils)
add_subdirectory(vulkan_raytracer)

add_subdirectory(editor)

# plugins
add_subdirectory(plugins)

# set editor as the start-up project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT editor)
