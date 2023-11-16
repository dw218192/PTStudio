#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "vulkan_raytracer")

#GLOB source files 
file(GLOB_RECURSE src_files 
	${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.*
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.*
)

set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
set(SHADERC_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SHADERC_SKIP_INSTALL ON CACHE BOOL "" FORCE)
set(SHADERC_DIR ${EXT_DIR}/shaderc)
find_package(PythonInterp 3 REQUIRED)

# compile shaderc manually because of debug vs release crt mismatch
# shaderc requires this python script to be run before building
execute_process(COMMAND ${PYTHON_EXECUTABLE} utils/git-sync-deps
    WORKING_DIRECTORY ${SHADERC_DIR}
)
set(SHADERC_ENABLE_SHARED_CRT ON CACHE BOOL "" FORCE)
add_subdirectory(${SHADERC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/shaderc)

find_package(Vulkan REQUIRED)
add_library(${lib_name} STATIC ${src_files})

target_include_directories(${lib_name} 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${GLFW_SOURCE_DIR}/include
)
# enable dynamic dispatch loader for vulkan hpp
target_compile_definitions(${lib_name} 
    PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC = 1
)

if (WIN32)
    target_compile_definitions(${lib_name} 
        PRIVATE
            NOMINMAX
            VK_USE_PLATFORM_WIN32_KHR
    )
endif()

target_link_libraries( ${lib_name}   
	$<BUILD_INTERFACE:developer_flags>
	core
    gl_wrapper
    Vulkan::Vulkan
    shaderc
)
