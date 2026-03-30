#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

using namespace pts::rendering;

TEST_CASE("profiler init" * doctest::test_suite("setup")) {
    PTS_STARTUP_PROFILER();
}

namespace {
auto make_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("dome_ibl_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("dome_ibl_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}
}  // namespace

TEST_CASE("env_texture_path defaults to empty") {
    LightData data;
    CHECK(data.env_texture_path.empty());
}

TEST_CASE("ibl_resources accessor returns same object") {
    RenderWorld world;
    auto& ibl = world.ibl_resources();
    CHECK_FALSE(ibl.is_ready());

    const auto& cworld = world;
    CHECK_FALSE(cworld.ibl_resources().is_ready());
}

#ifndef __EMSCRIPTEN__

TEST_CASE("update_ibl with no lights produces black uniform IBL") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;
    // Force a light version change so update_ibl processes
    {
        auto scope = world.begin_sync();
        // SyncScope destructor bumps light_version
    }

    world.update_ibl(device, device.queue());

    CHECK(world.ibl_resources().is_ready());
    CHECK(world.ibl_resources().brdf_lut_view() != nullptr);
    CHECK(world.ibl_resources().irradiance_view() != nullptr);
    CHECK(world.ibl_resources().prefiltered_env_view() != nullptr);
    CHECK(world.ibl_resources().env_cubemap_view() != nullptr);
    CHECK(world.ibl_resources().sampler() != nullptr);
}

TEST_CASE("update_ibl with dome light (no texture) produces uniform color IBL") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light_slot();
        auto w = scope.write_light(idx);
        w->type = LightData::Type::Dome;
        w->color = {1.0f, 0.9f, 0.8f};
        w->intensity = 0.3f;
    }

    world.update_ibl(device, device.queue());

    CHECK(world.ibl_resources().is_ready());
    CHECK(world.ibl_resources().irradiance_view() != nullptr);
    CHECK(world.ibl_resources().prefiltered_env_view() != nullptr);
    CHECK(world.ibl_resources().env_cubemap_view() != nullptr);
}

TEST_CASE("update_ibl skips when light_version unchanged") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light_slot();
        auto w = scope.write_light(idx);
        w->type = LightData::Type::Dome;
        w->color = {0.5f, 0.5f, 0.5f};
        w->intensity = 1.0f;
    }

    world.update_ibl(device, device.queue());
    CHECK(world.ibl_resources().is_ready());

    // Second call with no version change — should return early (no-op)
    world.update_ibl(device, device.queue());
    CHECK(world.ibl_resources().is_ready());
}

TEST_CASE("update_ibl transitions from dome to no-dome (black)") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;
    uint32_t dome_idx;
    {
        auto scope = world.begin_sync();
        dome_idx = scope.alloc_light_slot();
        auto w = scope.write_light(dome_idx);
        w->type = LightData::Type::Dome;
        w->color = {1.0f, 1.0f, 1.0f};
        w->intensity = 1.0f;
    }

    world.update_ibl(device, device.queue());
    CHECK(world.ibl_resources().is_ready());

    // Remove dome light
    {
        auto scope = world.begin_sync();
        scope.free_light_slot(dome_idx);
    }

    world.update_ibl(device, device.queue());
    // Still ready (black environment)
    CHECK(world.ibl_resources().is_ready());
}

TEST_CASE("update_ibl with Z-up produces ready IBL") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light_slot();
        auto w = scope.write_light(idx);
        w->type = LightData::Type::Dome;
        w->color = {1.0f, 1.0f, 1.0f};
        w->intensity = 1.0f;
    }

    world.update_ibl(device, device.queue(), UpAxis::Z);

    CHECK(world.ibl_resources().is_ready());
    CHECK(world.ibl_resources().irradiance_view() != nullptr);
    CHECK(world.ibl_resources().prefiltered_env_view() != nullptr);
    CHECK(world.ibl_resources().env_cubemap_view() != nullptr);
}

TEST_CASE("clear resets IBL state") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light_slot();
        auto w = scope.write_light(idx);
        w->type = LightData::Type::Dome;
        w->color = {1.0f, 1.0f, 1.0f};
        w->intensity = 1.0f;
    }
    world.update_ibl(device, device.queue());
    CHECK(world.ibl_resources().is_ready());

    world.clear();
    CHECK_FALSE(world.ibl_resources().is_ready());
}

#endif  // !__EMSCRIPTEN__
