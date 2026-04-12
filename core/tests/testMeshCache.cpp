#include <core/rendering/renderPass.h>
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
struct TestPass final : IPass {
    using IPass::IPass;
    auto name() const noexcept -> std::string_view override {
        return "test";
    }

    // Expose protected members for testing.
    using IPass::get_or_create_pass_data;
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

TEST_CASE("world swap invalidates pass data cache") {
    TestPass pass{s_test_sl};
    int factory_calls = 0;
    {
        RenderWorld world;
        auto scope = world.begin_sync();
        auto slot = scope.alloc_mesh_slot();
        {
            auto w = scope.write_mesh(slot);
            PTS_UNUSED(w);
        }
        pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot, world, [&]() {
            ++factory_calls;
            return 1;
        });
        CHECK(factory_calls == 1);
    }
    // Old world destroyed — cache gone. New world must recreate.
    RenderWorld world2;
    auto scope2 = world2.begin_sync();
    auto slot2 = scope2.alloc_mesh_slot();
    {
        auto w = scope2.write_mesh(slot2);
        PTS_UNUSED(w);
    }
    pass.get_or_create_pass_data<int>(PassDataKind::Mesh, slot2, world2, [&]() {
        ++factory_calls;
        return 99;
    });
    CHECK(factory_calls == 2);
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
