// pts_shaderc: build-time CLI wrapping run_slang() for WGSL compile.
//
// Usage:
//   pts_shaderc compile --source <file.slang> --output <file.wgsl>
//                       [-D DEFINE]... [-I DIR]...
//                       [--metadata <file.h> --namespace <ns>]
//                       [--force]

#include <core/rendering/shaderc/slangRuntime.h>
#include <slang-com-ptr.h>
#include <slang.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

using pts::rendering::run_slang;
using pts::rendering::SlangCompileOutput;

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "pts_shaderc: %s\n", msg.c_str());
    std::exit(1);
}

void print_usage() {
    std::fprintf(stderr,
                 "usage: pts_shaderc compile --source <file.slang> --output <file.wgsl>\n"
                 "                           [-D DEFINE]... [-I DIR]...\n"
                 "                           [--metadata <file.h> --namespace <ns>]\n"
                 "                           [--force]\n");
}

struct Args {
    std::filesystem::path source;
    std::filesystem::path output;
    std::filesystem::path metadata_output;
    std::string metadata_namespace;
    std::filesystem::path search_path;
    std::vector<std::string> defines;
    std::vector<std::string> entries;
    std::vector<std::filesystem::path> extra_search_paths;
    bool force = false;
};

Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        std::exit(1);
    }
    std::string_view verb = argv[1];
    if (verb == "-h" || verb == "--help") {
        print_usage();
        std::exit(0);
    }
    if (verb != "compile") {
        die("unknown verb '" + std::string(verb) + "' (only 'compile' supported)");
    }

    Args a;
    for (int i = 2; i < argc; ++i) {
        std::string_view v = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argc) die("missing value after " + std::string(v));
            return argv[i];
        };
        if (v == "--source") {
            a.source = std::filesystem::path(std::string(next()));
        } else if (v == "--output") {
            a.output = std::filesystem::path(std::string(next()));
        } else if (v == "--metadata") {
            a.metadata_output = std::filesystem::path(std::string(next()));
        } else if (v == "--namespace") {
            a.metadata_namespace = std::string(next());
        } else if (v == "-D") {
            a.defines.emplace_back(next());
        } else if (v == "-I" || v == "--search-path") {
            a.extra_search_paths.emplace_back(std::string(next()));
        } else if (v == "--entry") {
            a.entries.emplace_back(next());
        } else if (v == "--force" || v == "-f") {
            a.force = true;
        } else if (v == "-h" || v == "--help") {
            print_usage();
            std::exit(0);
        } else {
            die("unknown arg '" + std::string(v) + "'");
        }
    }
    if (a.source.empty()) die("missing --source");
    if (a.output.empty()) die("missing --output");
    if (!a.metadata_output.empty() && a.metadata_namespace.empty()) {
        die("--metadata requires --namespace");
    }

    if (!a.extra_search_paths.empty()) {
        a.search_path = a.extra_search_paths.front();
    }
    return a;
}

// -- Staleness check --
//
// Mirrors the pre-refactor slangc.py logic: rebuild when the output is
// missing, when the source or any sibling `.slang` module in the source
// directory has a newer mtime, or when any `.slang` file in a search path is
// newer. Also invalidates when a requested metadata header is absent or
// older than the WGSL output.
bool needs_compile(const std::filesystem::path& source, const std::filesystem::path& output,
                   const std::filesystem::path& metadata_output,
                   const std::vector<std::filesystem::path>& search_paths, bool force) {
    if (force) return true;
    std::error_code ec;
    if (!std::filesystem::exists(output, ec)) return true;
    auto out_mtime = std::filesystem::last_write_time(output, ec);
    if (ec) return true;

    auto scan_dir = [&](const std::filesystem::path& dir) -> bool {
        if (!std::filesystem::is_directory(dir, ec)) return false;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".slang") continue;
            auto mt = std::filesystem::last_write_time(entry.path(), ec);
            if (ec) continue;
            if (mt > out_mtime) return true;
        }
        return false;
    };

    if (scan_dir(source.parent_path())) return true;
    for (const auto& sp : search_paths) {
        if (scan_dir(sp)) return true;
    }

    if (!metadata_output.empty()) {
        if (!std::filesystem::exists(metadata_output, ec)) return true;
        auto md_mt = std::filesystem::last_write_time(metadata_output, ec);
        if (ec || md_mt < out_mtime) return true;
    }
    return false;
}

void write_text_atomic(const std::filesystem::path& path, std::string_view contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) die("failed to open output file: " + path.string());
    f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!f) die("failed to write output file: " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    if (!needs_compile(a.source, a.output, a.metadata_output, a.extra_search_paths, a.force)) {
        std::fprintf(stdout, "pts_shaderc: up-to-date %s\n", a.output.string().c_str());
        return 0;
    }

    Slang::ComPtr<slang::IGlobalSession> global_session;
    if (SLANG_FAILED(slang::createGlobalSession(global_session.writeRef())) || !global_session) {
        die("failed to create Slang global session");
    }

    std::vector<std::string_view> defines_view;
    defines_view.reserve(a.defines.size());
    for (const auto& d : a.defines) defines_view.emplace_back(d);

    SlangCompileOutput result = run_slang(global_session.get(), a.search_path, a.source, a.entries,
                                          defines_view, a.metadata_namespace);

    if (!result.diagnostics.empty()) {
        std::fwrite(result.diagnostics.data(), 1, result.diagnostics.size(), stderr);
        if (result.diagnostics.back() != '\n') std::fputc('\n', stderr);
    }
    if (!result.success) {
        die("compile failed");
    }

    write_text_atomic(a.output, result.wgsl);

    if (!a.metadata_output.empty()) {
        if (result.metadata_header.empty()) die("metadata emission failed");
        write_text_atomic(a.metadata_output, result.metadata_header);
    }
    return 0;
}
