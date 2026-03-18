#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/shaderLoader.h>
#include <doctest/doctest.h>
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

std::optional<std::string_view> missing_getter(std::string_view) {
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

TEST_CASE("ShaderLoader load delegates to embedded getter") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter);

    auto result = loader.load("shaders/test.wgsl");
    REQUIRE(result.has_value());
    CHECK(*result == "// embedded wgsl");
}

TEST_CASE("ShaderLoader load returns nullopt when embedded getter fails") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/missing.wgsl", "shaders/missing.slang",
                           "generated/shaders/missing.wgsl", missing_getter);

    auto result = loader.load("shaders/missing.wgsl");
    CHECK_FALSE(result.has_value());
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

    auto result_a = loader.load("shaders/a.wgsl");
    auto result_b = loader.load("shaders/b.wgsl");
    REQUIRE(result_a.has_value());
    REQUIRE(result_b.has_value());
    CHECK(*result_a == "// shader A");
    CHECK(*result_b == "// shader B");
}

TEST_CASE("ShaderLoader poll_and_reload returns empty with no dirty files") {
    ShaderLoader loader(make_logger());
    loader.register_shader("shaders/test.wgsl", "shaders/test.slang", "generated/shaders/test.wgsl",
                           fake_getter);

    auto changed = loader.poll_and_reload();
    CHECK(changed.empty());
}
