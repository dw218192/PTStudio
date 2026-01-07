#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

#Project name 
set(project_name "PTStudio")
project(${project_name} LANGUAGES CXX C)

# CUDA and C++ standard
set(CMAKE_CXX_STANDARD 17)

#default build type is Debug
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug)
endif ()

# options
# Use if(NOT DEFINED) to allow Conan cache variables to take precedence
if(NOT DEFINED PTSTUDIO_PATHTRACER)
    option(PTSTUDIO_PATHTRACER "Build CUDA pathtracer" OFF)
endif()
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
if (PTSTUDIO_PATHTRACER)
    include("cmake/AutoDetectCudaArch.cmake")
endif ()

# CUDA and C++ compiler flags
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

set(cuda_flags
    -std=c++17
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-Xcompiler -Wall -g -O0>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CONFIG:Debug>>>:-Xcompiler -Wall -O3>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:-Xcompiler -Wall -g -O0>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<CONFIG:Debug>>>:-Xcompiler -Wall -O3>
    $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler ${MSVC_XCOMPILER_FLAGS}>
    -Xcudafe=--display_error_number    
    -lineinfo    
    --expt-extended-lambda    
    -use_fast_math        
    --expt-relaxed-constexpr    
    -Xptxas -warn-spills -res-usage   
    --ptxas-options=-v    
)

# used to switch between cuda and cxx flags depending on the target
add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
# Only include CUDA toolkit directories if CUDA is available (when building path tracer)
if (PTSTUDIO_PATHTRACER)
    target_include_directories(developer_flags INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif ()

target_compile_features(developer_flags INTERFACE cxx_std_17)

set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext")

find_package(glfw3 REQUIRED)
find_package(glew REQUIRED)
find_package(glm REQUIRED)
find_package(tinyobjloader REQUIRED)
find_package(stb REQUIRED)
find_package(tcb-span REQUIRED)
find_package(fmt REQUIRED)
find_package(tl-expected REQUIRED)
find_package(nlohmann_json REQUIRED)

# external dependencies that are built from source
add_subdirectory(${EXT_DIR})

# add resource compiler
include(${EXT_DIR}/cmrc/CMakeRC.cmake)

# add subprojects
# add_subdirectory(core)

# renderers
add_subdirectory(gl_wrapper)
# if (PTSTUDIO_PATHTRACER)
#     add_subdirectory(cuda_pt_renderer)
# endif ()
# add_subdirectory(vulkan_raytracer)

# add_subdirectory(editor)

# # set editor as the start-up project
# set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT editor)
