#pragma once

// libslang-backed compile primitive -- native only. WASM builds never see this
// header (libslang isn't compiled for wasm in our pipeline).
#ifndef __EMSCRIPTEN__

#include <boost/core/span.hpp>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace slang {
struct IGlobalSession;
}  // namespace slang

namespace pts::rendering {

struct SlangCompileOutput {
    bool success = false;
    std::string wgsl;
    std::string metadata_header;  // populated when metadata_namespace is non-empty
    std::string cpp;
    std::string cpp_header;
    std::string types_header;
    std::vector<std::filesystem::path> dependencies;
    std::string diagnostics;
};

struct SlangCompileOptions {
    bool cpp = false;
    // Emit these named structs and dependencies using their WGSL buffer layouts.
    std::vector<std::string> type_names;
    std::string types_namespace;
};

/// Compile a single Slang source file via libslang. The default target is WGSL;
/// options.cpp generates portable C++ source and a header for host execution.
/// options.type_names selects named structs reflected in WGSL buffer layouts.
///
/// When `metadata_namespace` is non-empty, the linked reflection is walked
/// in-process and a C++ metadata header is written to `metadata_header`.
///
/// Shared by SlangCompiler (runtime) and pts_shaderc (build-time CLI).
/// Enforces column-major matrix layout and the canonical search path order
/// (source dir first, then the configured paths) to match slangc CLI defaults.
SlangCompileOutput run_slang(slang::IGlobalSession* global_session,
                             boost::span<const std::filesystem::path> search_paths,
                             const std::filesystem::path& slang_source,
                             const std::vector<std::string>& entry_points,
                             boost::span<const std::string_view> defines,
                             std::string_view metadata_namespace = {},
                             const SlangCompileOptions& options = {});

}  // namespace pts::rendering

#endif  // !__EMSCRIPTEN__
