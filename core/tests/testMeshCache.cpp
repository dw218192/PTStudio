#include <core/rendering/scenePass.h>

#include "testApplication.h"

using namespace pts::rendering;

namespace {

/// Concrete pass that exposes mesh_cache_get for testing.
struct TestPass final : IScenePass {
    auto name() const noexcept -> std::string_view override { return "test"; }
    auto is_ready() const noexcept -> bool override { return true; }
    void setup(const webgpu::Device& /*device*/) override {}
    void add_to_frame_graph(FrameGraph& /*fg*/, const PassContext& /*ctx*/) override {}

    // Expose protected members for testing.
    using IScenePass::mesh_cache_clear;
    using IScenePass::mesh_cache_get;
};

}  // namespace

TEST_CASE("mesh_cache_get creates entry on first call") {
    TestPass pass;
    int factory_calls = 0;
    auto& val = pass.mesh_cache_get<int>(0, 1, [&]() {
        ++factory_calls;
        return 42;
    });
    CHECK(val == 42);
    CHECK(factory_calls == 1);
}

TEST_CASE("mesh_cache_get returns cached value on same version") {
    TestPass pass;
    int factory_calls = 0;
    auto factory = [&]() {
        ++factory_calls;
        return 42;
    };
    pass.mesh_cache_get<int>(0, 1, factory);
    auto& val = pass.mesh_cache_get<int>(0, 1, factory);
    CHECK(val == 42);
    CHECK(factory_calls == 1);
}

TEST_CASE("mesh_cache_get re-creates on version change") {
    TestPass pass;
    int factory_calls = 0;
    pass.mesh_cache_get<int>(0, 1, [&]() {
        ++factory_calls;
        return 10;
    });
    auto& val = pass.mesh_cache_get<int>(0, 2, [&]() {
        ++factory_calls;
        return 20;
    });
    CHECK(val == 20);
    CHECK(factory_calls == 2);
}

TEST_CASE("mesh_cache_get supports different mesh indices") {
    TestPass pass;
    auto& a = pass.mesh_cache_get<int>(0, 1, []() { return 100; });
    auto& b = pass.mesh_cache_get<int>(1, 1, []() { return 200; });
    CHECK(a == 100);
    CHECK(b == 200);
}

TEST_CASE("mesh_cache_clear removes all entries") {
    TestPass pass;
    pass.mesh_cache_get<int>(0, 1, []() { return 1; });
    pass.mesh_cache_get<int>(1, 1, []() { return 2; });
    pass.mesh_cache_clear();

    int factory_calls = 0;
    pass.mesh_cache_get<int>(0, 1, [&]() {
        ++factory_calls;
        return 99;
    });
    CHECK(factory_calls == 1);
}

PTS_TEST_MAIN()
