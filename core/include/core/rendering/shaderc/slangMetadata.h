#pragma once

// libslang-backed C++ metadata-header emitter — native only. Walks a linked
// `slang::ShaderReflection` + `slang::IComponentType` and emits the
// `<name>_shader_metadata.h` file consumed by the C++ render passes
// (inline constexpr entry-point names, VertexLayout, create_bind_group_layout_N
// helpers, k_color_attachment_count). Replaces the Python shader_codegen.py
// Jinja template path — see the `pts_shaderc compile --metadata` flag.
//
// Byte-compat scope: reproduces the template output for the patterns in use
// today (constant buffers, varying inputs/outputs, single-element vertex
// vectors). Extend the walker rather than reintroducing a JSON detour.
#ifndef __EMSCRIPTEN__

#include <string>
#include <string_view>

namespace slang {
struct ShaderReflection;
struct IComponentType;
struct IGlobalSession;
}  // namespace slang

namespace pts::rendering {

/// Emit a C++ metadata header for the given linked reflection.
/// `ns` is the enclosing `namespace` name (single identifier). `target_index`
/// selects the Slang target (always 0 in our pipeline). `linked` may be null
/// in which case every binding is treated as used by every stage (permissive).
/// `global_session` is required for resolving user attributes (e.g.
/// `[DynamicBuffer]`); pass null only in contexts where attribute handling is
/// irrelevant.
std::string run_slang_metadata_header(slang::IGlobalSession* global_session,
                                      slang::ShaderReflection* reflection,
                                      slang::IComponentType* linked, std::string_view ns,
                                      int target_index = 0);

}  // namespace pts::rendering

#endif  // !__EMSCRIPTEN__
