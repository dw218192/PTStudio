#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "vulkan_raytracer")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
)
find_package(Vulkan REQUIRED)
add_library(${lib_name} STATIC ${src_files})

target_include_directories(${lib_name} 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${Vulkan_INCLUDE_DIRS}
        ${GLFW_SOURCE_DIR}/include
)

target_link_libraries( ${lib_name}   
	$<BUILD_INTERFACE:developer_flags>
	core
    ${Vulkan_LIBRARIES}
)
