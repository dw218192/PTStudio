# core_shaderc — shader-compile primitives shared between the runtime `core`
# library and the build-time `pts_shaderc` CLI.
#
# Intentionally a leaf target: only Slang / Boost-headers / spdlog. Keeping it
# out of the main `core` dep graph (no Dawn / ImGui / USD) means the host tool
# can link against it without dragging in the full engine.
#
# On Emscripten libslang isn't available — the runtime uses EmbeddedCompiler
# exclusively. Drop slang* translation units but keep ShaderLoader for the
# shared shader registry; diagnostics.h ships with core/include unconditionally.
#
# Public headers live at core/include/core/rendering/shaderc/ alongside the rest
# of core's public API.

set(_core_shaderc_sources shaderLoader.cpp)
if(NOT EMSCRIPTEN)
    list(APPEND _core_shaderc_sources
        slangRuntime.cpp
        slangMetadata.cpp
    )
endif()

add_library(core_shaderc STATIC ${_core_shaderc_sources})
add_library(core::shaderc ALIAS core_shaderc)

target_include_directories(core_shaderc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(core_shaderc PUBLIC
    Boost::headers
    spdlog::spdlog
)
if(NOT EMSCRIPTEN)
    target_link_libraries(core_shaderc PUBLIC slang::slang)
endif()
