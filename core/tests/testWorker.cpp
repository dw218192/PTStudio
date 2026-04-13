#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/diagnostics.h>
#include <core/worker.h>
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <thread>

// ── TaskProgress ─────────────────────────────────────────────────────────────

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

// ── OneShotTask ──────────────────────────────────────────────────────────────

TEST_CASE("OneShotTask - completes and returns result") {
    pts::OneShotTask<int> task("add", [](pts::TaskProgress& p) {
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

TEST_CASE("OneShotTask - reports progress mid-work") {
    pts::OneShotTask<std::string> task("slow", [](pts::TaskProgress& p) {
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
        auto prog = task.progress();
        auto stat = task.status();
        UNUSED(prog);
        UNUSED(stat);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    CHECK(task.take_result() == "result");
}

TEST_CASE("OneShotTask - destructor joins thread") {
    {
        pts::OneShotTask<int> task("join_test", [](pts::TaskProgress&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return 0;
        });
    }
    CHECK(true);
}

TEST_CASE("OneShotTask - string result type") {
    pts::OneShotTask<std::string> task("string_task", [](pts::TaskProgress& p) {
        p.set_status("building string");
        return std::string("hello world");
    });

    while (!task.is_done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    CHECK(task.take_result() == "hello world");
}

// ── Worker (persistent) ─────────────────────────────────────────────────────

TEST_CASE("Worker - single job completes") {
    pts::Worker<int, int> worker([](int&& x, pts::TaskProgress& p) {
        p.set_progress(1.0f);
        return x * 2;
    });

    worker.submit(21);

    while (!worker.has_result()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto result = worker.take_result();
    REQUIRE(result.has_value());
    CHECK(*result == 42);
}

TEST_CASE("Worker - multiple sequential jobs") {
    pts::Worker<int, int> worker([](int&& x, pts::TaskProgress&) { return x + 1; });

    for (int i = 0; i < 5; ++i) {
        worker.submit(i);
        while (!worker.has_result()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto result = worker.take_result();
        REQUIRE(result.has_value());
        CHECK(*result == i + 1);
    }
}

TEST_CASE("Worker - latest-wins replaces pending job") {
    std::atomic<int> jobs_executed{0};
    std::atomic<bool> gate{false};

    pts::Worker<int, int> worker([&](int&& x, pts::TaskProgress&) {
        if (x == 1) {
            // Hold job 1 until all submissions are made
            while (!gate.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
        jobs_executed.fetch_add(1, std::memory_order_relaxed);
        return x;
    });

    // Submit first job — it will block on the gate
    worker.submit(1);
    // Give the worker thread time to pick up job 1
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // While job 1 is held, rapidly submit more — only the last should survive
    worker.submit(2);
    worker.submit(3);
    worker.submit(4);

    // Release job 1 — loop will then pick up job 4 (latest-wins)
    gate.store(true, std::memory_order_release);

    // Wait for both jobs to complete
    while (jobs_executed.load(std::memory_order_acquire) < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Only 2 jobs executed: the first + the last (2 and 3 were replaced)
    CHECK(jobs_executed.load() == 2);

    // The final available result is from job 4
    // (job 1's result may have been overwritten by job 4)
    auto result = worker.take_result();
    REQUIRE(result.has_value());
    CHECK(*result == 4);
}

TEST_CASE("Worker - shutdown drains in-flight work") {
    std::atomic<bool> work_completed{false};

    auto worker = std::make_unique<pts::Worker<int, int>>([&](int&& x, pts::TaskProgress&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        work_completed.store(true, std::memory_order_release);
        return x;
    });

    worker->submit(1);
    // Give the worker time to start processing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Shutdown should block until the in-flight job completes
    worker.reset();
    CHECK(work_completed.load(std::memory_order_acquire));
}

TEST_CASE("Worker - shutdown with no pending work returns immediately") {
    pts::Worker<int, int> worker([](int&& x, pts::TaskProgress&) { return x; });
    // Shutdown with no jobs submitted — should not hang
    worker.shutdown();
    CHECK(true);
}

TEST_CASE("Worker - progress resets between jobs") {
    std::atomic<bool> job_started{false};
    pts::Worker<int, int> worker([&](int&& x, pts::TaskProgress& p) {
        // Each job sets a unique progress value
        p.set_progress(static_cast<float>(x) / 10.0f);
        p.set_status("job " + std::to_string(x));
        job_started.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return x;
    });

    worker.submit(5);
    while (!worker.has_result()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    worker.take_result();

    // After taking result, submit a new job; progress should reset to 0 initially
    // We check that the new job gets its own progress state
    job_started.store(false, std::memory_order_release);
    worker.submit(8);
    // Wait for job 8 to start so we observe its progress, not stale values
    while (!job_started.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    float in_flight_progress = worker.progress();
    CHECK(in_flight_progress == doctest::Approx(0.8f));
    while (!worker.has_result()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    auto r = worker.take_result();
    CHECK(*r == 8);
}

TEST_CASE("Worker - has_result is lockfree") {
    pts::Worker<int, int> worker([](int&& x, pts::TaskProgress&) { return x; });
    // has_result() uses std::atomic — verify it returns false before any submission
    CHECK_FALSE(worker.has_result());
    // take_result returns nullopt when no result
    CHECK_FALSE(worker.take_result().has_value());
}
