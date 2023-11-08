#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "cuda_pt_renderer")

enable_language(CUDA)

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
)

#Dependency
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

#Executable
cuda_add_library(${lib_name} ${src_files})
set_property(TARGET ${lib_name} PROPERTY 
	CUDA_SEPARABLE_COMPILATION ON)

#Libs linked to the executable
target_link_libraries( ${lib_name}   
	$<BUILD_INTERFACE:developer_flags>
	core
	gl_wrapper
	Thrust
)
