#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "vulkan_raytracer")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
)
file(GLOB_RECURSE res_files 
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.*
)

add_library(${lib_name} STATIC ${src_files})

target_include_directories(${lib_name} 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_raytracer
)

if (WIN32)
    target_compile_definitions(${lib_name} 
        PRIVATE
            NOMINMAX
            VK_USE_PLATFORM_WIN32_KHR
    )
endif()

target_compile_definitions(${lib_name}
    PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
)

# add resources
cmrc_add_resource_library(${lib_name}_resources
	ALIAS ${lib_name}::rc
	${res_files}
)

target_link_libraries( ${lib_name}
    PRIVATE
        $<BUILD_INTERFACE:developer_flags>
    PUBLIC
        core
        gl_utils
        glfw
        Vulkan::Headers
        Vulkan::Vulkan
        shaderc::shaderc
        ${lib_name}::rc
)
