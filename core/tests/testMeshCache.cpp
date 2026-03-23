#include <core/rendering/scenePass.h>
#include <core/rendering/shaderLoader.h>
#include <spdlog/spdlog.h>

#include "testApplication.h"

using namespace pts;
using namespace pts::rendering;

namespace {

static ShaderLoader make_test_shader_loader() {
    return ShaderLoader(spdlog::default_logger());
}
static auto s_test_sl = make_test_shader_loader();

/// Concrete pass that exposes get_or_create_pass_data for testing.
struct TestPass final : IScenePass {
    using IScenePass::IScenePass;
    auto name() const noexcept -> std::string_view override {
        return "test";
    }
    auto is_ready() const noexcept -> bool override {
        return true;
    }
    void do_setup(const webgpu::Device& /*device*/) override {
    }
    void add_to_frame_graph(FrameGraph& /*fg*/, const PassContext& /*ctx*/) override {
    }

    // Expose protected members for testing.
    using IScenePass::clear_pass_data;
    using IScenePass::get_or_create_pass_data;
};

}  // namespace

TEST_CASE("get_or_create_pass_data creates entry on first call") {
    TestPass pass{s_test_sl};
    RenderWorld world;
    auto scope = world.begin_sync();
    auto slot = scope.alloc_mesh_slot();
    {
        auto w = scope.write_mesh(slot);
        PTS_UNUSED(w);
    }

    int factory_calls = 0;
    auto& val = pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, [&]() {
        ++factory_calls;
        return 42;
    });
    CHECK(val == 42);
    CHECK(factory_calls == 1);
}

TEST_CASE("get_or_create_pass_data returns cached value on same version") {
    TestPass pass{s_test_sl};
    RenderWorld world;
    auto scope = world.begin_sync();
    auto slot = scope.alloc_mesh_slot();
    {
        auto w = scope.write_mesh(slot);
        PTS_UNUSED(w);
    }

    int factory_calls = 0;
    auto factory = [&]() {
        ++factory_calls;
        return 42;
    };
    pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, factory);
    auto& val = pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, factory);
    CHECK(val == 42);
    CHECK(factory_calls == 1);
}

TEST_CASE("get_or_create_pass_data re-creates on version change") {
    TestPass pass{s_test_sl};
    RenderWorld world;
    uint32_t slot;
    {
        auto scope = world.begin_sync();
        slot = scope.alloc_mesh_slot();
        {
            auto w = scope.write_mesh(slot);
            PTS_UNUSED(w);
        }
    }

    int factory_calls = 0;
    pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, [&]() {
        ++factory_calls;
        return 10;
    });

    // Bump mesh generation via write guard
    {
        auto scope = world.begin_sync();
        auto w = scope.write_mesh(slot);
        PTS_UNUSED(w);
    }

    auto& val = pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, [&]() {
        ++factory_calls;
        return 20;
    });
    CHECK(val == 20);
    CHECK(factory_calls == 2);
}

TEST_CASE("get_or_create_pass_data supports different keys") {
    TestPass pass{s_test_sl};
    RenderWorld world;
    uint32_t s0, s1;
    {
        auto scope = world.begin_sync();
        s0 = scope.alloc_mesh_slot();
        s1 = scope.alloc_mesh_slot();
        // Bump generation on each via write guard
        {
            auto w = scope.write_mesh(s0);
            PTS_UNUSED(w);
        }
        {
            auto w = scope.write_mesh(s1);
            PTS_UNUSED(w);
        }
    }
    auto& a =
        pass.get_or_create_pass_data<int>(PassDataKind::Mesh, s0, world, []() { return 100; });
    auto& b =
        pass.get_or_create_pass_data<int>(PassDataKind::Mesh, s1, world, []() { return 200; });
    CHECK(a == 100);
    CHECK(b == 200);
}

TEST_CASE("clear_pass_data removes all entries") {
    TestPass pass{s_test_sl};
    RenderWorld world;
    auto scope = world.begin_sync();
    auto slot = scope.alloc_mesh_slot();
    {
        auto w = scope.write_mesh(slot);
        PTS_UNUSED(w);
    }

    pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, []() { return 1; });
    pass.clear_pass_data();

    int factory_calls = 0;
    pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, [&]() {
        ++factory_calls;
        return 99;
    });
    CHECK(factory_calls == 1);
}

TEST_CASE("get_or_create_pass_data with nullptr factory succeeds on hit") {
    TestPass pass{s_test_sl};
    RenderWorld world;
    auto scope = world.begin_sync();
    auto slot = scope.alloc_mesh_slot();
    {
        auto w = scope.write_mesh(slot);
        PTS_UNUSED(w);
    }

    pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, []() { return 42; });
    auto& val = pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, nullptr);
    CHECK(val == 42);
}

PTS_TEST_MAIN()
