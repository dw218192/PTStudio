#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/backgroundTask.h>
#include <core/diagnostics.h>
#include <doctest/doctest.h>

#include <chrono>
#include <thread>

TEST_CASE("TaskProgress - default state") {
    pts::TaskProgress progress;
    CHECK(progress.progress() == doctest::Approx(0.0f));
    CHECK(progress.status().empty());
}

TEST_CASE("TaskProgress - set and read progress") {
    pts::TaskProgress progress;
    progress.set_progress(0.5f);
    CHECK(progress.progress() == doctest::Approx(0.5f));
}

TEST_CASE("TaskProgress - set and read status") {
    pts::TaskProgress progress;
    progress.set_status("loading");
    CHECK(progress.status() == "loading");
}

TEST_CASE("BackgroundTask - completes and returns result") {
    pts::BackgroundTask<int> task("add", [](pts::TaskProgress& p) {
        p.set_progress(1.0f);
        p.set_status("done");
        return 42;
    });

    while (!task.is_done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    CHECK(task.is_done());
    CHECK(task.name() == "add");
    CHECK(task.progress() == doctest::Approx(1.0f));
    CHECK(task.status() == "done");
    CHECK(task.take_result() == 42);
}

TEST_CASE("BackgroundTask - reports progress mid-work") {
    pts::BackgroundTask<std::string> task("slow", [](pts::TaskProgress& p) {
        p.set_progress(0.0f);
        p.set_status("starting");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        p.set_progress(0.5f);
        p.set_status("halfway");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        p.set_progress(1.0f);
        return std::string("result");
    });

    while (!task.is_done()) {
        // Reading progress/status while running should not crash
        auto prog = task.progress();
        auto stat = task.status();
        PTS_UNUSED(prog);
        PTS_UNUSED(stat);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    CHECK(task.take_result() == "result");
}

TEST_CASE("BackgroundTask - destructor joins thread") {
    // Should not hang or crash
    {
        pts::BackgroundTask<int> task("join_test", [](pts::TaskProgress&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return 0;
        });
    }
    CHECK(true);
}

TEST_CASE("BackgroundTask - string result type") {
    pts::BackgroundTask<std::string> task("string_task", [](pts::TaskProgress& p) {
        p.set_status("building string");
        return std::string("hello world");
    });

    while (!task.is_done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    CHECK(task.take_result() == "hello world");
}
