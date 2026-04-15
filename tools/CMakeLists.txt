# Standalone entry point for the --host-tools-only CI build.
#
# This file is NEVER added as a subdirectory of the root project -- the root
# declares pts_shaderc inline with the rest of its native targets. This file
# exists only so that Linux CI can build pts_shaderc in isolation, with just
# slang + boost + spdlog from tools/conanfile.txt, skipping the full runtime
# dep graph (Dawn, USD, GLFW/xorg).

cmake_minimum_required(VERSION 3.19)
project(pts_host_tools CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED)
find_package(spdlog REQUIRED)
find_package(slang REQUIRED)

# PTS_CORE_SHADERC_DIR lets the caller point at a vendored copy of the
# core/shaderc tree. Defaults to the sibling path in a normal checkout.
if(NOT PTS_CORE_SHADERC_DIR)
    set(PTS_CORE_SHADERC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../core/shaderc")
endif()
if(NOT EXISTS "${PTS_CORE_SHADERC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "host-tools standalone build: core/shaderc not found at ${PTS_CORE_SHADERC_DIR}. "
                        "Pass -DPTS_CORE_SHADERC_DIR=<path> or check out the full repo.")
endif()
add_subdirectory("${PTS_CORE_SHADERC_DIR}" core_shaderc_build)

# pts_shaderc -- Slang -> WGSL compiler CLI used by the slangc prebuild step.
# Shares the run_slang() backend with runtime SlangCompiler via core::shaderc.
add_executable(pts_shaderc pts_shaderc/main.cpp)
target_link_libraries(pts_shaderc PRIVATE core::shaderc)
