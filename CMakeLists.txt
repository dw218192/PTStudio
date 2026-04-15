#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

#Project name 
set(project_name "PTStudio")
project(${project_name} LANGUAGES CXX C)

# Include custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#default build type is Debug
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug)
endif ()

# options
# Use if(NOT DEFINED) to allow Conan cache variables to take precedence
if(NOT DEFINED CORE_BUILD_TESTS)
    option(CORE_BUILD_TESTS "Build core unit tests" ON)
endif()
if(NOT DEFINED PTS_WINDOWING)
    set(PTS_WINDOWING "glfw")
endif()

if(EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
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
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
    $<$<NOT:$<BOOL:${EMSCRIPTEN}>>:PTS_WORKSPACE_ROOT="${CMAKE_SOURCE_DIR}">
)

target_compile_features(developer_flags INTERFACE cxx_std_17)

find_package(glm REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
if(EMSCRIPTEN)
    find_package(Boost REQUIRED)
else()
    find_package(Boost REQUIRED COMPONENTS filesystem stacktrace stacktrace_from_exception)
endif()
find_package(cxxopts REQUIRED)
find_package(openusd CONFIG REQUIRED)

# Optional USD schema modules (beyond the always-linked usd_usd core).
# Populated from config.yaml via the build tool's -DPTS_USD_MODULES flag.
set(PTS_USD_MODULES "usdGeom;usdShade;usdLux;geomUtil;hd" CACHE STRING
    "Optional USD schema modules to link")

# On Emscripten, embed OpenUSD plugin resources into the WASM virtual filesystem
# so the Plug registry can discover types at runtime.  Conan's CMakeDeps does not
# propagate the --embed-file link options from pxrTargets, so we replicate them
# here on openusd::usd_plug (transitively linked by every USD consumer).
# Embeds: top-level plugInfo.json, plus each module's resources/ directory
# (plugInfo.json + generatedSchema.usda).
if(EMSCRIPTEN)
    if(DEFINED openusd_PACKAGE_FOLDER_DEBUG)
        set(_pxr_usd_dir "${openusd_PACKAGE_FOLDER_DEBUG}/lib/usd")
    elseif(DEFINED openusd_PACKAGE_FOLDER_RELEASE)
        set(_pxr_usd_dir "${openusd_PACKAGE_FOLDER_RELEASE}/lib/usd")
    endif()
    if(DEFINED _pxr_usd_dir AND EXISTS "${_pxr_usd_dir}/plugInfo.json")
        set(_embed_flags "SHELL:--embed-file ${_pxr_usd_dir}/plugInfo.json@/usd/plugInfo.json")
        file(GLOB _pxr_res_dirs LIST_DIRECTORIES true "${_pxr_usd_dir}/*/resources")
        foreach(_res_dir ${_pxr_res_dirs})
            get_filename_component(_module_dir "${_res_dir}" DIRECTORY)
            get_filename_component(_module_name "${_module_dir}" NAME)
            list(APPEND _embed_flags
                 "SHELL:--embed-file ${_res_dir}@/usd/${_module_name}/resources")
        endforeach()
        set_property(TARGET openusd::usd_plug APPEND PROPERTY
                     INTERFACE_LINK_OPTIONS ${_embed_flags})
    endif()
    # Force --whole-archive on USD static archives whose ARCH_CONSTRUCTOR
    # registrations (TfType, schema factories, resolvers, plugin discovery)
    # would otherwise be dead-stripped by the linker.
    set(_pxr_whole_archive_libs usd_plug usd_tf usd_sdf usd_ar usd_vt usd_gf)
    foreach(_mod IN LISTS PTS_USD_MODULES)
        list(APPEND _pxr_whole_archive_libs usd_${_mod})
    endforeach()
    list(REMOVE_DUPLICATES _pxr_whole_archive_libs)
    foreach(_lib IN LISTS _pxr_whole_archive_libs)
        find_library(_pxr_${_lib} NAMES ${_lib}
                     PATHS "${_pxr_usd_dir}/.." NO_DEFAULT_PATH)
        if(_pxr_${_lib})
            set_property(TARGET openusd::${_lib} APPEND PROPERTY
                         INTERFACE_LINK_OPTIONS
                         "SHELL:-Wl,--whole-archive ${_pxr_${_lib}} -Wl,--no-whole-archive")
        endif()
        unset(_pxr_${_lib} CACHE)
    endforeach()
endif()

if (NOT EMSCRIPTEN)
    if (PTS_WINDOWING STREQUAL "glfw")
        find_package(glfw3 REQUIRED)
    endif()

    find_package(Dawn REQUIRED)
    find_package(portable-file-dialogs REQUIRED)
    find_package(Tracy REQUIRED)
    find_package(slang REQUIRED)
endif()

# GUI libraries from Conan
find_package(imgui REQUIRED)
find_package(imguizmo REQUIRED)
find_package(stb REQUIRED)
find_package(OpenEXR REQUIRED)


# Emscripten configuration - centralized flags for all targets
# Must be defined before add_subdirectory() so that targets can link to it.
add_library(wasm_flags INTERFACE)
if(EMSCRIPTEN)
    # emdawnwebgpu port: the build tool downloads the remote port file matching
    # the pinned Dawn version and passes it via -DEMDAWNWEBGPU_PORT_FILE=.
    if(NOT EMDAWNWEBGPU_PORT_FILE)
        message(FATAL_ERROR "EMDAWNWEBGPU_PORT_FILE must be set for Emscripten builds. "
            "Use the build tool (./repo build --platform emscripten) which downloads the correct version.")
    endif()
    target_compile_options(wasm_flags INTERFACE
        "--use-port=${EMDAWNWEBGPU_PORT_FILE}"
        "--use-port=contrib.glfw3"
        "-pthread"
        "-fexceptions")
    target_link_options(wasm_flags INTERFACE
        "--use-port=${EMDAWNWEBGPU_PORT_FILE}"
        "--use-port=contrib.glfw3"
        "-pthread"
        "-sPTHREAD_POOL_SIZE=(typeof navigator!=='undefined'?navigator.hardwareConcurrency:4)||4"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sMAXIMUM_MEMORY=4GB"
        "-sINITIAL_MEMORY=512MB"
        "-fexceptions"
        "$<$<CONFIG:Release>:--closure=1>"
        "--shell-file=${CMAKE_SOURCE_DIR}/web/index.html"
        "--emrun")
endif()

# Shared shader-compile primitives (in-tree library, linked by both `core`
# and the `pts_shaderc` host-tool CLI).
add_subdirectory(core/shaderc)

# Host tools used by prebuild steps (native only).
# For --host-tools-only CI builds, tools/CMakeLists.txt provides a standalone
# entry point that builds pts_shaderc without the rest of the project graph.
if(NOT EMSCRIPTEN)
    add_executable(usdz_pack tools/conan/usdz_pack/usdzPack.cpp)
    target_link_libraries(usdz_pack PRIVATE openusd::usd_usdUtils)

    add_executable(pts_shaderc tools/pts_shaderc/main.cpp)
    target_link_libraries(pts_shaderc PRIVATE core::shaderc)
endif()

# add subprojects
add_subdirectory(core)
add_subdirectory(renderers)
add_subdirectory(hello_triangle)
add_subdirectory(editor)

# set editor as the start-up project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT editor)
