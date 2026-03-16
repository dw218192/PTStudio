#include <core/rendering/scenePass.h>

#include "testApplication.h"

using namespace pts;
using namespace pts::rendering;

namespace {

/// Concrete pass that exposes cache_get for testing.
struct TestPass final : IScenePass {
    auto name() const noexcept -> std::string_view override {
        return "test";
    }
    auto is_ready() const noexcept -> bool override {
        return true;
    }
    void setup(const webgpu::Device& /*device*/) override {
    }
    void add_to_frame_graph(FrameGraph& /*fg*/, const PassContext& /*ctx*/) override {
    }

    // Expose protected members for testing.
    using IScenePass::cache_clear;
    using IScenePass::cache_get;
};

}  // namespace

TEST_CASE("cache_get creates entry on first call") {
    TestPass pass;
    int factory_calls = 0;
    auto& val = pass.cache_get<int>(0, 1, [&]() {
        ++factory_calls;
        return 42;
    });
    CHECK(val == 42);
    CHECK(factory_calls == 1);
}

TEST_CASE("cache_get returns cached value on same version") {
    TestPass pass;
    int factory_calls = 0;
    auto factory = [&]() {
        ++factory_calls;
        return 42;
    };
    pass.cache_get<int>(0, 1, factory);
    auto& val = pass.cache_get<int>(0, 1, factory);
    CHECK(val == 42);
    CHECK(factory_calls == 1);
}

TEST_CASE("cache_get re-creates on version change") {
    TestPass pass;
    int factory_calls = 0;
    pass.cache_get<int>(0, 1, [&]() {
        ++factory_calls;
        return 10;
    });
    auto& val = pass.cache_get<int>(0, 2, [&]() {
        ++factory_calls;
        return 20;
    });
    CHECK(val == 20);
    CHECK(factory_calls == 2);
}

TEST_CASE("cache_get supports different keys") {
    TestPass pass;
    auto& a = pass.cache_get<int>(0, 1, []() { return 100; });
    auto& b = pass.cache_get<int>(1, 1, []() { return 200; });
    CHECK(a == 100);
    CHECK(b == 200);
}

TEST_CASE("cache_clear removes all entries") {
    TestPass pass;
    pass.cache_get<int>(0, 1, []() { return 1; });
    pass.cache_get<int>(1, 1, []() { return 2; });
    pass.cache_clear();

    int factory_calls = 0;
    pass.cache_get<int>(0, 1, [&]() {
        ++factory_calls;
        return 99;
    });
    CHECK(factory_calls == 1);
}

TEST_CASE("cache_get with nullptr factory asserts on miss") {
    TestPass pass;
    // Populate cache first
    pass.cache_get<int>(0, 1, []() { return 42; });
    // Hit with nullptr — should succeed
    auto& val = pass.cache_get<int>(0, 1, nullptr);
    CHECK(val == 42);
}

PTS_TEST_MAIN()
