#pragma once

#define DOCTEST_CONFIG_IMPLEMENT
#include <core/application.h>
#include <core/loggingManager.h>
#include <doctest/doctest.h>

namespace pts::test {

namespace detail {
/// Holds Application dependencies, initialized before the Application base.
struct TestDeps {
    pts::Config config{};
    pts::LoggingManager logging_manager{config};
};
}  // namespace detail

/**
 * @brief Minimal test harness derived from Application.
 *
 * Runs doctest::Context in the first frame, then stops.  Because it inherits
 * from Application it participates in the Emscripten event loop (needed for
 * WASM tests with PROXY_TO_PTHREAD).
 *
 * Usage — each test translation unit:
 *   #include "testApplication.h"
 *   TEST_CASE("...") { ... }
 *   PTS_TEST_MAIN()
 */
struct TestApplication : private detail::TestDeps, public pts::Application {
    TestApplication() : detail::TestDeps{}, Application("test", logging_manager) {
    }

    void loop(float /*dt*/) override {
        doctest::Context ctx;
        m_result = ctx.run();
        request_stop();
    }

    [[nodiscard]] int result() const noexcept {
        return m_result;
    }

   private:
    int m_result{0};
};

inline int run_tests() {
    TestApplication app;
    app.run();
    return app.result();
}

}  // namespace pts::test

/// Provide main() for a test executable using TestApplication.
#define PTS_TEST_MAIN()                \
    int main() {                       \
        return pts::test::run_tests(); \
    }
