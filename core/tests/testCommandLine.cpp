#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/commandLine.h>
#include <doctest/doctest.h>

#include <array>
#include <string>

// Helper: build a mutable argv from string literals.
// cxxopts expects non-const char*, so we cast away const
// (it never actually writes through the pointers).
template <std::size_t N>
static auto make_argv(const char* const (&args)[N]) -> std::array<char*, N> {
    std::array<char*, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = const_cast<char*>(args[i]);
    }
    return out;
}

TEST_CASE("CommandLine - Parse --help returns false") {
    pts::CommandLine cli;
    const char* args[] = {"app", "--help"};
    auto argv = make_argv(args);

    CHECK(cli.parse(2, argv.data()) == false);
}

TEST_CASE("CommandLine - Parse string option") {
    pts::CommandLine cli;
    cli.add_string("log-level", "log level");

    const char* args[] = {"app", "--log-level", "debug"};
    auto argv = make_argv(args);

    REQUIRE(cli.parse(3, argv.data()) == true);
    CHECK(cli.get_string("log-level") == "debug");
    CHECK(cli.has("log-level") == true);
}

TEST_CASE("CommandLine - Parse flag option") {
    pts::CommandLine cli;
    cli.add_flag("verbose", "verbose output");

    SUBCASE("Flag present") {
        const char* args[] = {"app", "--verbose"};
        auto argv = make_argv(args);

        REQUIRE(cli.parse(2, argv.data()) == true);
        CHECK(cli.get_flag("verbose") == true);
    }

    SUBCASE("Flag absent") {
        const char* args[] = {"app"};
        auto argv = make_argv(args);

        REQUIRE(cli.parse(1, argv.data()) == true);
        CHECK(cli.get_flag("verbose") == false);
    }
}

TEST_CASE("CommandLine - has() for absent option") {
    pts::CommandLine cli;
    cli.add_string("level", "log level");

    const char* args[] = {"app"};
    auto argv = make_argv(args);

    REQUIRE(cli.parse(1, argv.data()) == true);
    CHECK(cli.has("level") == false);
    CHECK(cli.has("nonexistent") == false);
}

TEST_CASE("CommandLine - Unknown args do not fail") {
    pts::CommandLine cli;

    const char* args[] = {"app", "--unknown-flag", "value"};
    auto argv = make_argv(args);

    CHECK(cli.parse(3, argv.data()) == true);
}

TEST_CASE("CommandLine - add_string with default value") {
    pts::CommandLine cli;
    cli.add_string("level", "log level", std::string("info"));

    const char* args[] = {"app"};
    auto argv = make_argv(args);

    REQUIRE(cli.parse(1, argv.data()) == true);
    // has() should be false — the option was not explicitly provided on the command line
    CHECK_FALSE(cli.has("level"));
    CHECK(cli.get_string("level") == "info");
}

TEST_CASE("CommandLine - Multiple options") {
    pts::CommandLine cli;
    cli.add_string("log-level", "log level");
    cli.add_string("plugins-dir", "plugins directory");
    cli.add_flag("quit-on-start", "quit after start");

    const char* args[] = {"app",           "--log-level", "warn",
                          "--plugins-dir", "my_plugins",  "--quit-on-start"};
    auto argv = make_argv(args);

    REQUIRE(cli.parse(6, argv.data()) == true);
    CHECK(cli.get_string("log-level") == "warn");
    CHECK(cli.get_string("plugins-dir") == "my_plugins");
    CHECK(cli.get_flag("quit-on-start") == true);
    CHECK(cli.has("log-level") == true);
    CHECK(cli.has("plugins-dir") == true);
}

TEST_CASE("CommandLine - add_string with implicit_value") {
    pts::CommandLine cli;
    cli.add_string("output", "output path", std::nullopt, std::string(""));

    SUBCASE("present without value uses implicit_value") {
        const char* args[] = {"app", "--output"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(2, argv.data()) == true);
        CHECK(cli.has("output") == true);
        CHECK(cli.get_string("output") == "");
    }

    SUBCASE("present with value uses provided value") {
        const char* args[] = {"app", "--output=foo.png"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(2, argv.data()) == true);
        CHECK(cli.has("output") == true);
        CHECK(cli.get_string("output") == "foo.png");
    }

    SUBCASE("absent returns call-site default") {
        const char* args[] = {"app"};
        auto argv = make_argv(args);
        REQUIRE(cli.parse(1, argv.data()) == true);
        CHECK(cli.has("output") == false);
        CHECK(cli.get_string("output", "fallback") == "fallback");
    }
}

TEST_CASE("CommandLine - get_string fallback default_value") {
    pts::CommandLine cli;
    cli.add_string("level", "log level");

    const char* args[] = {"app"};
    auto argv = make_argv(args);

    REQUIRE(cli.parse(1, argv.data()) == true);
    CHECK(cli.get_string("level", "fallback") == "fallback");
}
