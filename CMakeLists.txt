#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

#Project name 
set(project_name "PTStudio")
project(${project_name} LANGUAGES CXX C)

# Include custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PluginHelpers)

#default build type is Debug
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug)
endif ()

# options
# Use if(NOT DEFINED) to allow Conan cache variables to take precedence
if(NOT DEFINED CORE_BUILD_TESTS)
    option(CORE_BUILD_TESTS "Build core unit tests" ON)
endif()
if(NOT DEFINED PTS_STATIC_PLUGINS)
    option(PTS_STATIC_PLUGINS "Build plugins as static libraries" OFF)
endif()
if(NOT DEFINED PTS_WINDOWING)
    set(PTS_WINDOWING "glfw")
endif()

if(EMSCRIPTEN)
    set(PTS_STATIC_PLUGINS ON CACHE BOOL "" FORCE)
endif()

# Direct all output to /bin directory
# Since we organize builds by configuration (_build/Debug, _build/Release),
# we don't want CMake to append the config name again
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
# Prevent multi-config generators from adding config suffix
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# C++ compiler flags
set(cxx_flags
	$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /MP>
	$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/MDd /Zi /Ob0 /Od /RTC1>
	# Add MSVC-specific compiler flags here
	
	$<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -O3 -fmax-errors=1>
	$<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-g -O0>
	# Add GCC compiler flags here

	$<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<BOOL:${EMSCRIPTEN}>>>:-Wall -m64 -O3 -ferror-limit=1>
	$<$<AND:$<CXX_COMPILER_ID:Clang>,$<BOOL:${EMSCRIPTEN}>>:-Wall -O3 -ferror-limit=1>
	$<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:-g -O0>
	# Add Clang compiler flags here
)

add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
)
target_compile_definitions(developer_flags INTERFACE
    PTS_WINDOWING="${PTS_WINDOWING}"
    "PTS_WINDOWING_${PTS_WINDOWING}=1"
)

target_compile_features(developer_flags INTERFACE cxx_std_17)

find_package(glm REQUIRED)
find_package(tinyobjloader REQUIRED)
find_package(stb REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
if(EMSCRIPTEN)
    find_package(Boost REQUIRED)
else()
    find_package(Boost REQUIRED COMPONENTS filesystem)
endif()
find_package(openusd CONFIG REQUIRED)

if (NOT EMSCRIPTEN)
    if (PTS_WINDOWING STREQUAL "glfw")
        find_package(glfw3 REQUIRED)
    endif()

    find_package(Dawn REQUIRED)
    find_package(portable-file-dialogs REQUIRED)
endif()

# GUI libraries from Conan
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)
find_package(imgui_color_text_edit REQUIRED)


# Emscripten configuration - centralized flags for all targets
# Must be defined before add_subdirectory() so that targets can link to it.
add_library(wasm_flags INTERFACE)
if(EMSCRIPTEN)
    # emdawnwebgpu port: the build tool downloads the remote port file matching
    # the pinned Dawn version and passes it via -DEMDAWNWEBGPU_PORT_FILE=.
    if(NOT EMDAWNWEBGPU_PORT_FILE)
        message(FATAL_ERROR "EMDAWNWEBGPU_PORT_FILE must be set for Emscripten builds. "
            "Use the build tool (./pts build --platform emscripten) which downloads the correct version.")
    endif()
    target_compile_options(wasm_flags INTERFACE
        "--use-port=${EMDAWNWEBGPU_PORT_FILE}"
        "--use-port=contrib.glfw3")
    target_link_options(wasm_flags INTERFACE
        "--use-port=${EMDAWNWEBGPU_PORT_FILE}"
        "--use-port=contrib.glfw3"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sMAXIMUM_MEMORY=4GB"
        "-sINITIAL_MEMORY=512MB"
        "$<$<CONFIG:Release>:--closure=1>"
        "--shell-file=${CMAKE_SOURCE_DIR}/web/index.html")
endif()

# add subprojects
add_subdirectory(core)
add_subdirectory(hello_triangle)
add_subdirectory(editor)

# plugins
add_subdirectory(plugins)

# Link static plugins into the main executable when enabled.
if(PTS_STATIC_PLUGINS)
    target_link_libraries(editor PRIVATE pts::plugins)
endif()

# set editor as the start-up project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT editor)
