// pts_shaderc: build-time Slang -> WGSL/C++ compiler and GPU layout generator.
//
// Usage:
//   pts_shaderc compile --source <file.slang> --output <file.wgsl>
//                       [-D DEFINE]... [-I DIR]...
//                       [--metadata <file.h> --namespace <ns>]
//                       [--cpp-header <file.h>]
//                       [--types-header <file.h> --types-namespace <ns>
//                        --type <name> ...]
//                       [--force]

#include <core/rendering/shaderc/slangRuntime.h>
#include <slang-com-ptr.h>
#include <slang.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
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
                 "                           [--cpp-header <file.h>] (output becomes C++)\n"
                 "                           [--types-header <file.h> --types-namespace <ns>\n"
                 "                            --type <name> ...]\n"
                 "                           [--force]\n");
}

struct Args {
    std::filesystem::path source;
    std::filesystem::path output;
    std::filesystem::path metadata_output;
    std::filesystem::path cpp_header;
    std::filesystem::path types_header;
    pts::rendering::SlangCompileOptions options;
    std::string metadata_namespace;
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
        } else if (v == "--cpp-header") {
            a.cpp_header = std::string(next());
            a.options.cpp = true;
        } else if (v == "--types-header") {
            a.types_header = std::string(next());
        } else if (v == "--types-namespace") {
            a.options.types_namespace = std::string(next());
        } else if (v == "--type") {
            a.options.type_names.emplace_back(next());
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
    if ((!a.types_header.empty() || !a.options.type_names.empty() ||
         !a.options.types_namespace.empty()) &&
        (a.types_header.empty() || a.options.type_names.empty() ||
         a.options.types_namespace.empty())) {
        die("type generation requires --types-header, --types-namespace and --type");
    }
    if (a.options.cpp && (!a.metadata_output.empty() || !a.types_header.empty()))
        die("C++ generation cannot emit GPU metadata");
    std::vector<std::filesystem::path> outputs{a.output};
    for (const auto& path : {a.metadata_output, a.cpp_header, a.types_header}) {
        if (path.empty()) continue;
        for (const auto& other : outputs) {
            if (std::filesystem::absolute(path).lexically_normal() ==
                std::filesystem::absolute(other).lexically_normal())
                die("output paths must be distinct");
        }
        outputs.push_back(path);
    }

    return a;
}

// -- Staleness check --
//
// Track Slang's transitive dependencies, including shared C++/Slang headers.
// Also scan module directories to detect changes in import resolution.
std::filesystem::path dependency_file(const std::filesystem::path& output) {
    auto path = output;
    path += ".deps";
    return path;
}

bool needs_compile(const Args& args, std::string_view signature) {
    const auto& source = args.source;
    const auto& output = args.output;
    const auto& search_paths = args.extra_search_paths;
    const bool force = args.force;
    if (force) return true;
    std::error_code ec;
    if (!std::filesystem::exists(output, ec)) return true;
    auto out_mtime = std::filesystem::last_write_time(output, ec);
    if (ec) return true;
    std::ifstream signature_file(output.string() + ".args", std::ios::binary);
    std::string previous((std::istreambuf_iterator<char>(signature_file)), {});
    if (!signature_file || previous != signature) return true;

    std::ifstream dependencies(dependency_file(output));
    if (!dependencies) return true;
    std::string dependency;
    bool has_dependencies = false;
    while (std::getline(dependencies, dependency)) {
        if (dependency.empty()) return true;
        has_dependencies = true;
        auto mt = std::filesystem::last_write_time(dependency, ec);
        if (ec || mt > out_mtime) return true;
    }
    if (dependencies.bad() || !has_dependencies) return true;

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

    for (const auto& additional_output :
         {args.metadata_output, args.cpp_header, args.types_header}) {
        if (additional_output.empty()) continue;
        if (!std::filesystem::exists(additional_output, ec)) return true;
        auto md_mt = std::filesystem::last_write_time(additional_output, ec);
        if (ec || md_mt < out_mtime) return true;
    }
    return false;
}

void write_text_atomic(const std::filesystem::path& path, std::string_view contents) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) die("failed to open output file: " + path.string());
    f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!f) die("failed to write output file: " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    Slang::ComPtr<slang::IGlobalSession> global_session;
    if (SLANG_FAILED(slang::createGlobalSession(global_session.writeRef())) || !global_session) {
        die("failed to create Slang global session");
    }

    // Options and compiler changes invalidate every output, including generated
    // C++ and upload headers. The executable itself is a transitive dependency.
    std::ostringstream signature;
    signature << global_session->getBuildTagString() << '\n';
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) != "--force" && std::string_view(argv[i]) != "-f")
            signature << std::string_view(argv[i]).size() << ':' << argv[i] << '\n';
    }
    if (!needs_compile(a, signature.str())) {
        std::fprintf(stdout, "pts_shaderc: up-to-date %s\n", a.output.string().c_str());
        return 0;
    }

    std::vector<std::string_view> defines_view;
    defines_view.reserve(a.defines.size());
    for (const auto& d : a.defines) defines_view.emplace_back(d);

    SlangCompileOutput result = run_slang(global_session.get(), a.extra_search_paths, a.source,
                                          a.entries, defines_view, a.metadata_namespace, a.options);

    if (!result.diagnostics.empty()) {
        std::fwrite(result.diagnostics.data(), 1, result.diagnostics.size(), stderr);
        if (result.diagnostics.back() != '\n') std::fputc('\n', stderr);
    }
    if (!result.success) {
        die("compile failed");
    }

    write_text_atomic(a.output, a.options.cpp ? result.cpp : result.wgsl);
    if (!a.cpp_header.empty()) write_text_atomic(a.cpp_header, result.cpp_header);
    if (!a.types_header.empty()) write_text_atomic(a.types_header, result.types_header);

    if (!a.metadata_output.empty()) {
        if (result.metadata_header.empty()) die("metadata emission failed");
        write_text_atomic(a.metadata_output, result.metadata_header);
    }
    std::string dependencies;
    result.dependencies.emplace_back(std::filesystem::absolute(argv[0]));
    result.dependencies.emplace_back(std::filesystem::absolute(a.source));
    for (const auto& path : result.dependencies) {
        dependencies += std::filesystem::absolute(path).generic_string() + '\n';
    }
    write_text_atomic(dependency_file(a.output), dependencies);
    write_text_atomic(a.output.string() + ".args", signature.str());
    return 0;
}
