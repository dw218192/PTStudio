#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

#Project name 
set(project_name "PTStudio")
project(${project_name} LANGUAGES CXX C CUDA)  

#default build type is Debug
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug)
endif ()

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Auto-detect GPU architecture
include("cmake/AutoDetectCudaArch.cmake")

# CUDA and C++ compiler flags
set(cxx_flags 
	$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /std:c++17>
	$<$<CONFIG:Debug>:/MDd /Zi /Ob0 /Od /RTC1>
	# Add MSVC-specific compiler flags here
	
	$<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -O3 -std=c++17>
	$<$<CONFIG:Debug>:-g -O0>
	# Add GCC compiler flags here
	
	$<$<CXX_COMPILER_ID:Clang>:-Wall -m64 -O3 -std=c++17>
	$<$<CONFIG:Debug>:-g -O0>
	# Add Clang compiler flags here
)

set(MSVC_XCOMPILER_FLAGS
	"/std:c++17"
	$<$<CONFIG:Debug>:/MDd /Zi /Ob0 /Od /RTC1>
)

set(cuda_flags
    #$<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-Xcompiler -Wall -g -O0>
    #$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CONFIG:Debug>>>:-Xcompiler -Wall -O3>
    #$<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:-Xcompiler -Wall -g -O0>
    #$<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<CONFIG:Debug>>>:-Xcompiler -Wall -O3>
    # $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler ${MSVC_XCOMPILER_FLAGS}>
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
target_include_directories(developer_flags INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_features(developer_flags INTERFACE cxx_std_17)

enable_testing()

add_subdirectory(pathtracer)