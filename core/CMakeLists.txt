cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(lib_name "core")

# ============================================================================
# imgui_backends - compiles imgui backends using Conan's imgui package
# ============================================================================
# The core imgui library comes from Conan (imgui::imgui), but backends are 
# provided as source files in res/bindings/ and must be compiled by the consumer

# Get imgui package folder from Conan (set by CMakeDeps generator).
# Multi-config generators (VS, Xcode) leave CMAKE_BUILD_TYPE empty at
# configure time, so probe all known configs and pick the first defined.
if(CMAKE_CONFIGURATION_TYPES)
    foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        if(DEFINED imgui_PACKAGE_FOLDER_${_cfg})
            set(IMGUI_PKG_ROOT "${imgui_PACKAGE_FOLDER_${_cfg}}")
            break()
        endif()
    endforeach()
else()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
    set(IMGUI_PKG_ROOT "${imgui_PACKAGE_FOLDER_${_build_type_upper}}")
endif()
if(NOT IMGUI_PKG_ROOT)
    message(FATAL_ERROR "Unable to resolve imgui package folder from Conan CMakeDeps")
endif()
set(IMGUI_BINDINGS_DIR "${IMGUI_PKG_ROOT}/res/bindings")

# Collect backend sources based on configuration
set(IMGUI_BACKEND_SOURCES)
if(PTS_WINDOWING STREQUAL "glfw")
    list(APPEND IMGUI_BACKEND_SOURCES
        ${IMGUI_BINDINGS_DIR}/imgui_impl_glfw.cpp
        ${IMGUI_BINDINGS_DIR}/imgui_impl_glfw.h
    )
endif()
# Add WebGPU backend
list(APPEND IMGUI_BACKEND_SOURCES
    ${IMGUI_BINDINGS_DIR}/imgui_impl_wgpu.cpp
    ${IMGUI_BINDINGS_DIR}/imgui_impl_wgpu.h
)

add_library(imgui_backends STATIC ${IMGUI_BACKEND_SOURCES})
target_include_directories(imgui_backends PUBLIC ${IMGUI_BINDINGS_DIR})
target_link_libraries(imgui_backends PUBLIC imgui::imgui)
target_compile_definitions(imgui_backends PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_DAWN)
if(PTS_WINDOWING STREQUAL "glfw")
    target_link_libraries(imgui_backends PRIVATE glfw)
endif()
if(EMSCRIPTEN)
    target_link_libraries(imgui_backends PUBLIC wasm_flags)
else()
    target_link_libraries(imgui_backends PUBLIC dawn::webgpu_dawn)
endif()

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

file(GLOB_RECURSE core_webgpu_sources
	${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/webgpu/*.*
)
list(APPEND core_rendering_sources ${core_webgpu_sources})

file(GLOB_RECURSE core_adapter_sources
	${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/adapters/*.*
)
list(APPEND core_rendering_sources ${core_adapter_sources})

set(src_files ${core_src_files} ${core_rendering_sources})
list(REMOVE_DUPLICATES src_files)

#Executable
add_library(${lib_name} STATIC ${src_files})

# WebGPU uses [0,1] depth; tell GLM to generate matching projection matrices
target_compile_definitions(${lib_name} PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

#Libs linked to the executable
target_link_libraries(${lib_name}
PRIVATE
	$<BUILD_INTERFACE:developer_flags>
	stb::stb
PUBLIC
	openusd::usd_usd
	glm::glm
	imgui::imgui
	imgui_backends
	fmt::fmt
	spdlog::spdlog
	nlohmann_json::nlohmann_json
	TBB::tbb
)

# Optional USD schema modules (configured via PTS_USD_MODULES)
foreach(_mod IN LISTS PTS_USD_MODULES)
	target_link_libraries(${lib_name} PUBLIC openusd::usd_${_mod})
endforeach()

if (NOT EMSCRIPTEN)
	target_link_libraries(${lib_name}
		PRIVATE portable-file-dialogs::portable-file-dialogs
		PUBLIC dawn::webgpu_dawn Boost::filesystem Tracy::TracyClient
		PRIVATE cxxopts::cxxopts
		       Boost::stacktrace Boost::stacktrace_from_exception
		       slang::slang
	)
else()
	target_link_libraries(${lib_name} PUBLIC wasm_flags Boost::boost PRIVATE cxxopts::cxxopts)
endif()

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
PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/include/core
)

# Build unit tests if enabled
if (CORE_BUILD_TESTS)
	add_subdirectory(tests)
endif()