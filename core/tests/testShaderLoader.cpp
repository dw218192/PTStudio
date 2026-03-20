#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/shaderLoader.h>
#include <doctest/doctest.h>
#include <generated/embedded_test_resources.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

using namespace pts::rendering;

namespace {

std::optional<std::string_view> fake_getter(std::string_view key) {
    if (key == "shaders/test.wgsl") {
        return "// embedded wgsl";
    }
    return std::nullopt;
}

std::shared_ptr<spdlog::logger> make_logger() {
    auto logger = spdlog::get("shader_loader_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("shader_loader_test");
    }
    return logger;
}

}  // namespace

TEST_CASE("ShaderLoader load returns embedded content after registration") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter);

    auto result = loader.load("shaders/test.wgsl");
    CHECK(result == "// embedded wgsl");
}

TEST_CASE("ShaderLoader supports multiple independent shader registrations") {
    ShaderLoader loader(make_logger());

    auto getter_a = [](std::string_view key) -> std::optional<std::string_view> {
        if (key == "shaders/a.wgsl") return "// shader A";
        return std::nullopt;
    };
    auto getter_b = [](std::string_view key) -> std::optional<std::string_view> {
        if (key == "shaders/b.wgsl") return "// shader B";
        return std::nullopt;
    };

    loader.register_shader("shaders/a.wgsl", "shaders/a.slang", "generated/a.wgsl", getter_a);
    loader.register_shader("shaders/b.wgsl", "shaders/b.slang", "generated/b.wgsl", getter_b);

    CHECK(loader.load("shaders/a.wgsl") == "// shader A");
    CHECK(loader.load("shaders/b.wgsl") == "// shader B");
}

TEST_CASE("ShaderLoader poll_and_reload returns empty with no dirty files") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter);

    auto changed = loader.poll_and_reload();
    CHECK(changed.empty());
}

TEST_CASE("ShaderLoader poll_and_start_reload returns false with no dirty files") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter);

    CHECK_FALSE(loader.poll_and_start_reload());
    CHECK_FALSE(loader.is_reloading());
}

TEST_CASE("ShaderLoader try_finish_reload returns empty when no task started") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter);

    auto changed = loader.try_finish_reload();
    CHECK(changed.empty());
}

TEST_CASE("ShaderLoader is_reloading returns false when no task started") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter);

    CHECK_FALSE(loader.is_reloading());
}

TEST_CASE("ShaderLoader register_shader accepts custom entry_points") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter, {"cs_main"});

    auto result = loader.load("shaders/test.wgsl");
    CHECK(result == "// embedded wgsl");
}

#ifdef PTS_SHADER_HOT_RELOAD
TEST_CASE("ShaderLoader discovers dependencies from real slang file") {
    ShaderLoader loader(make_logger());

    // Use the real test shader — simple.slang has only vertex_main
    loader.register_shader("shaders/test/simple.wgsl", "assets/shaders/test/simple.slang",
                           "assets/shaders/test/simple.wgsl", test_resources::get_resource,
                           {"vertex_main"});

    // Embedded content is served correctly
    auto result = loader.load("shaders/test/simple.wgsl");
    CHECK_FALSE(result.empty());

    // No dirty files after initial registration
    CHECK_FALSE(loader.poll_and_start_reload());
}
#endif
