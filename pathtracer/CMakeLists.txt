#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "pathtracer")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*)

#Executable
add_executable(${exec_name} ${src_files})

set_property(TARGET ${exec_name} PROPERTY 
	CUDA_SEPARABLE_COMPILATION ON)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

#Libs linked to the executable
target_link_libraries( ${exec_name}   
	$<BUILD_INTERFACE:developer_flags>
	Thrust
)

