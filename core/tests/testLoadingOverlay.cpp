#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/imgui/loadingOverlay.h>
#include <doctest/doctest.h>

TEST_CASE("LoadingOverlay - track adds active task") {
    pts::LoadingOverlay overlay;
    overlay.track(
        {"Test Task", [] { return false; }, [] { return 0.5f; }, [] { return "working"; }});
    CHECK(overlay.has_active_tasks());
}

TEST_CASE("LoadingOverlay - has_active_tasks reflects tracked state") {
    pts::LoadingOverlay overlay;
    CHECK_FALSE(overlay.has_active_tasks());

    bool done = false;
    overlay.track({"Task", [&] { return done; }, [] { return 0.0f; }, [] { return ""; }});
    CHECK(overlay.has_active_tasks());
}

TEST_CASE("LoadingOverlay - TrackedTask lambdas are type-erased") {
    pts::LoadingOverlay overlay;

    float progress = 0.0f;
    std::string status = "init";
    bool done = false;

    overlay.track({
        "Erased Task",
        [&] { return done; },
        [&] { return progress; },
        [&] { return status; },
    });

    CHECK(overlay.has_active_tasks());

    progress = 0.75f;
    status = "almost";
    done = true;

    // Verify lambdas capture correctly -- task reports done now
    // After next draw() call it would be pruned, but has_active_tasks
    // doesn't prune (only draw does), so it still shows active.
    CHECK(overlay.has_active_tasks());
}

TEST_CASE("LoadingOverlay - multiple tasks tracked") {
    pts::LoadingOverlay overlay;
    overlay.track({"A", [] { return false; }, [] { return 0.1f; }, [] { return "a"; }});
    overlay.track({"B", [] { return false; }, [] { return 0.2f; }, [] { return "b"; }});
    overlay.track({"C", [] { return false; }, [] { return 0.3f; }, [] { return "c"; }});
    CHECK(overlay.has_active_tasks());
}
