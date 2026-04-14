#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/diagnostics.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/slangCompiler.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

using namespace pts::rendering;
namespace fs = std::filesystem;

namespace {

std::shared_ptr<spdlog::logger> test_logger() {
    auto logger = spdlog::get("slang_compiler_test");
    if (!logger) logger = spdlog::stdout_color_mt("slang_compiler_test");
    return logger;
}

// Stub getter used for SlangCompiler tests — the real compiler output replaces
// this content; we only need register_shader's embedded_getter precondition
// satisfied.
std::optional<std::string_view> stub_getter(std::string_view /*key*/) {
    return std::string_view{"// stub wgsl"};
}

fs::path unique_cache_dir(const char* tag) {
    auto base = fs::temp_directory_path() / "pts_slang_cache_test";
    auto dir = base / (std::string(tag) + "_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

fs::path write_temp_slang(const fs::path& dir, const std::string& name,
                          const std::string& contents) {
    fs::create_directories(dir);
    auto path = dir / name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return path;
}

struct SlangFixture {
    fs::path tmp_dir;
    fs::path cache_dir;
    fs::path workspace_root;
    ShaderLoader loader;
    std::unique_ptr<SlangCompiler> compiler;

    explicit SlangFixture(const char* tag) : loader(test_logger()) {
        tmp_dir = unique_cache_dir(tag);
        cache_dir = tmp_dir / "cache";
        workspace_root = tmp_dir / "workspace";
        fs::create_directories(workspace_root);
        compiler = std::make_unique<SlangCompiler>(loader, test_logger(), cache_dir, workspace_root,
                                                   workspace_root, nullptr);
    }
    ~SlangFixture() {
        compiler.reset();
        std::error_code ec;
        fs::remove_all(tmp_dir, ec);
    }
};

constexpr const char* k_simple_slang = R"(
struct VSIn { float3 pos : POSITION; };
struct VSOut { float4 pos : SV_Position; };
[shader("vertex")]
VSOut vs_main(VSIn i) {
    VSOut o; o.pos = float4(i.pos, 1.0); return o;
}
)";

}  // namespace

TEST_CASE("SlangCompiler compiles a simple slang file to WGSL") {
    SlangFixture f("basic");
    auto path = write_temp_slang(f.workspace_root, "simple.slang", k_simple_slang);
    f.loader.register_shader("test/simple.wgsl", "simple.slang", "test/simple.wgsl", stub_getter,
                             {"vs_main"});

    auto wgsl = f.compiler->compile(ShaderKey{"test/simple.wgsl"});
    CHECK_FALSE(wgsl.empty());

    // Disk cache populated
    bool found_wgsl = false;
    for (auto& entry : fs::directory_iterator(f.cache_dir)) {
        if (entry.path().extension() == ".wgsl") {
            found_wgsl = true;
            break;
        }
    }
    CHECK(found_wgsl);
}

TEST_CASE("SlangCompiler second compile hits cache (same output, no file timestamp change)") {
    SlangFixture f("cache_hit");
    write_temp_slang(f.workspace_root, "simple.slang", k_simple_slang);
    f.loader.register_shader("test/simple.wgsl", "simple.slang", "test/simple.wgsl", stub_getter,
                             {"vs_main"});

    auto wgsl1 = f.compiler->compile(ShaderKey{"test/simple.wgsl"});
    auto wgsl2 = f.compiler->compile(ShaderKey{"test/simple.wgsl"});
    CHECK(wgsl1 == wgsl2);
}

TEST_CASE("SlangCompiler poll_dirty returns empty with no changes") {
    SlangFixture f("poll_clean");
    write_temp_slang(f.workspace_root, "simple.slang", k_simple_slang);
    f.loader.register_shader("test/simple.wgsl", "simple.slang", "test/simple.wgsl", stub_getter,
                             {"vs_main"});

    UNUSED(f.compiler->compile(ShaderKey{"test/simple.wgsl"}));
    CHECK(f.compiler->poll_dirty().empty());
}

TEST_CASE("SlangCompiler poll_dirty detects source file mtime change") {
    SlangFixture f("poll_dirty");
    auto path = write_temp_slang(f.workspace_root, "simple.slang", k_simple_slang);
    f.loader.register_shader("test/simple.wgsl", "simple.slang", "test/simple.wgsl", stub_getter,
                             {"vs_main"});

    auto r0 = f.compiler->source_revision("test/simple.wgsl");
    UNUSED(f.compiler->compile(ShaderKey{"test/simple.wgsl"}));

    // Advance mtime: filesystem timestamps on some filesystems have 1-second
    // granularity, so sleep briefly, then touch the file.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    {
        std::ofstream f_out(path, std::ios::binary | std::ios::app);
        f_out << "\n// touched\n";
    }

    auto dirty = f.compiler->poll_dirty();
    REQUIRE(dirty.size() == 1);
    CHECK(dirty[0] == "test/simple.wgsl");
    CHECK(f.compiler->source_revision("test/simple.wgsl") > r0);
}

TEST_CASE("SlangCompiler poll_dirty flags all dependents when shared header changes") {
    SlangFixture f("dep_capture");
    // Shared header, two shaders include it.
    auto header = write_temp_slang(f.workspace_root, "shared.slang",
                                   "float4 tint() { return float4(1.0, 0.5, 0.25, 1.0); }\n");
    write_temp_slang(f.workspace_root, "a.slang", std::string(R"(
#include "shared.slang"
struct VSIn { float3 pos : POSITION; };
struct VSOut { float4 pos : SV_Position; float4 col; };
[shader("vertex")]
VSOut vs_main(VSIn i) { VSOut o; o.pos = float4(i.pos, 1.0); o.col = tint(); return o; }
)"));
    write_temp_slang(f.workspace_root, "b.slang", std::string(R"(
#include "shared.slang"
struct VSIn { float3 pos : POSITION; };
struct VSOut { float4 pos : SV_Position; float4 col; };
[shader("vertex")]
VSOut vs_main(VSIn i) { VSOut o; o.pos = float4(i.pos, 1.0); o.col = tint() * 0.5; return o; }
)"));

    f.loader.register_shader("test/a.wgsl", "a.slang", "test/a.wgsl", stub_getter, {"vs_main"});
    f.loader.register_shader("test/b.wgsl", "b.slang", "test/b.wgsl", stub_getter, {"vs_main"});

    UNUSED(f.compiler->compile(ShaderKey{"test/a.wgsl"}));
    UNUSED(f.compiler->compile(ShaderKey{"test/b.wgsl"}));

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    {
        std::ofstream f_out(header, std::ios::binary | std::ios::app);
        f_out << "// touched\n";
    }

    auto dirty = f.compiler->poll_dirty();
    CHECK(dirty.size() == 2);
    bool has_a = false, has_b = false;
    for (auto& k : dirty) {
        if (k == "test/a.wgsl") has_a = true;
        if (k == "test/b.wgsl") has_b = true;
    }
    CHECK(has_a);
    CHECK(has_b);
}

TEST_CASE("SlangCompiler concurrent compile for same key does not corrupt disk cache") {
    SlangFixture f("concurrent");
    write_temp_slang(f.workspace_root, "simple.slang", k_simple_slang);
    f.loader.register_shader("test/simple.wgsl", "simple.slang", "test/simple.wgsl", stub_getter,
                             {"vs_main"});

    constexpr int k_threads = 8;
    std::vector<std::thread> threads;
    std::vector<std::string> results(k_threads);
    std::atomic<int> started{0};
    for (int i = 0; i < k_threads; ++i) {
        threads.emplace_back([&, i]() {
            ++started;
            while (started.load() < k_threads) {
                std::this_thread::yield();
            }
            results[i] = f.compiler->compile(ShaderKey{"test/simple.wgsl"});
        });
    }
    for (auto& t : threads) t.join();

    for (int i = 1; i < k_threads; ++i) {
        CHECK(results[i] == results[0]);
    }
    CHECK_FALSE(results[0].empty());
}
