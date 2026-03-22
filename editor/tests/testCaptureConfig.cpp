#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/commandLine.h>
#include <doctest/doctest.h>

#include <array>
#include <string>

// Minimal reproduction of AppConfig to test without pulling in all editor deps
namespace test {
struct AppConfig {
    std::string capture_output;
    std::string usd_path;
    std::string usd_override_path;
    int capture_frames = 1;
    std::string renderer_name;
    std::string debug_output_name;

    [[nodiscard]] bool is_capture_mode() const {
        return !capture_output.empty();
    }
};
}  // namespace test

template <std::size_t N>
static auto make_argv(const char* const (&args)[N]) -> std::array<char*, N> {
    std::array<char*, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = const_cast<char*>(args[i]);
    }
    return out;
}

TEST_CASE("AppConfig - is_capture_mode") {
    SUBCASE("default is not capture mode") {
        test::AppConfig cfg;
        CHECK_FALSE(cfg.is_capture_mode());
    }

    SUBCASE("non-empty capture_output enables capture mode") {
        test::AppConfig cfg;
        cfg.capture_output = "output.png";
        CHECK(cfg.is_capture_mode());
    }
}

TEST_CASE("AppConfig - default values") {
    test::AppConfig cfg;
    CHECK(cfg.capture_output.empty());
    CHECK(cfg.usd_path.empty());
    CHECK(cfg.usd_override_path.empty());
    CHECK(cfg.capture_frames == 1);
    CHECK(cfg.renderer_name.empty());
    CHECK(cfg.debug_output_name.empty());
}

TEST_CASE("Capture CLI flags registration and parsing") {
    pts::CommandLine cli;
    cli.add_string("capture-and-quit", "Render, capture viewport to PNG, then quit", std::nullopt,
                   std::string(""));
    cli.add_string("usd", "Load USD file instead of embedded default scene", std::nullopt);
    cli.add_string("usd-override", "Apply override layer on top of loaded scene", std::nullopt);
    cli.add_int("frames", "Frames to render before capture", 1);
    cli.add_string("renderer", "Select renderer by name", std::nullopt);
    cli.add_string("debug-output", "Capture debug target instead of scene_color", std::nullopt);

    SUBCASE("no capture flags -> not capture mode") {
        const char* args[] = {"editor"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(1, argv.data()));
        CHECK_FALSE(cli.has("capture-and-quit"));
    }

    SUBCASE("--capture-and-quit without path generates default") {
        const char* args[] = {"editor", "--capture-and-quit"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(2, argv.data()));
        CHECK(cli.has("capture-and-quit"));
        CHECK(cli.get_string("capture-and-quit") == "");
    }

    SUBCASE("--capture-and-quit with path (= syntax)") {
        const char* args[] = {"editor", "--capture-and-quit=out.png"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(2, argv.data()));
        CHECK(cli.has("capture-and-quit"));
        CHECK(cli.get_string("capture-and-quit") == "out.png");
    }

    SUBCASE("all flags together") {
        const char* args[] = {"editor",         "--capture-and-quit=capture.png",
                              "--usd",          "scene.usda",
                              "--usd-override", "override.usda",
                              "--frames",       "5",
                              "--renderer",     "Wireframe",
                              "--debug-output", "Normals"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(12, argv.data()));
        CHECK(cli.get_string("capture-and-quit") == "capture.png");
        CHECK(cli.get_string("usd") == "scene.usda");
        CHECK(cli.get_string("usd-override") == "override.usda");
        CHECK(cli.get_int("frames") == 5);
        CHECK(cli.get_string("renderer") == "Wireframe");
        CHECK(cli.get_string("debug-output") == "Normals");
    }

    SUBCASE("--frames defaults to 1") {
        const char* args[] = {"editor"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(1, argv.data()));
        CHECK(cli.get_int("frames") == 1);
    }
}
