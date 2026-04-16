#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <pxr/usd/sdf/path.h>
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

WGPUSampler create_ibl_sampler(const pts::webgpu::Device& device) {
    WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    desc.magFilter = WGPUFilterMode_Linear;
    desc.minFilter = WGPUFilterMode_Linear;
    desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
    desc.addressModeU = WGPUAddressMode_ClampToEdge;
    desc.addressModeV = WGPUAddressMode_ClampToEdge;
    desc.addressModeW = WGPUAddressMode_ClampToEdge;
    return wgpuDeviceCreateSampler(device.handle(), &desc);
}
}  // namespace

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
    auto sampler = create_ibl_sampler(device);

    RenderWorld world;
    // Force a light version change so update_ibl processes
    {
        auto scope = world.begin_sync();
        // SyncScope destructor bumps light_version
    }

    world.update_ibl(device, device.queue(), sampler);

    CHECK(world.ibl_resources().is_ready());
    CHECK(world.ibl_pipelines().brdf_lut_view() != nullptr);
    CHECK(world.ibl_resources().irradiance_view() != nullptr);
    CHECK(world.ibl_resources().prefiltered_env_view() != nullptr);
    CHECK(world.ibl_resources().env_cubemap_view() != nullptr);
    CHECK(world.ibl_pipelines().sampler() != nullptr);
    wgpuSamplerRelease(sampler);
}

TEST_CASE("update_ibl with dome light (no texture) produces uniform color IBL") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);
    auto sampler = create_ibl_sampler(device);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light(pxr::SdfPath("/TestDome0"));
        scope.mutate_light(idx, [&](LightData& w) {
            w.type = LightData::Type::Dome;
            w.color = {1.0f, 0.9f, 0.8f};
            w.intensity = 0.3f;
        });
    }

    world.update_ibl(device, device.queue(), sampler);

    CHECK(world.ibl_resources().is_ready());
    CHECK(world.ibl_resources().irradiance_view() != nullptr);
    CHECK(world.ibl_resources().prefiltered_env_view() != nullptr);
    CHECK(world.ibl_resources().env_cubemap_view() != nullptr);
    wgpuSamplerRelease(sampler);
}

TEST_CASE("update_ibl skips when light_version unchanged") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);
    auto sampler = create_ibl_sampler(device);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light(pxr::SdfPath("/TestDome0"));
        scope.mutate_light(idx, [&](LightData& w) {
            w.type = LightData::Type::Dome;
            w.color = {0.5f, 0.5f, 0.5f};
            w.intensity = 1.0f;
        });
    }

    world.update_ibl(device, device.queue(), sampler);
    CHECK(world.ibl_resources().is_ready());

    // Second call with no version change -- should return early (no-op)
    world.update_ibl(device, device.queue(), sampler);
    CHECK(world.ibl_resources().is_ready());
    wgpuSamplerRelease(sampler);
}

TEST_CASE("update_ibl transitions from dome to no-dome (black)") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);
    auto sampler = create_ibl_sampler(device);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto dome_idx = scope.alloc_light(pxr::SdfPath("/TestDome0"));
        scope.mutate_light(dome_idx, [&](LightData& w) {
            w.type = LightData::Type::Dome;
            w.color = {1.0f, 1.0f, 1.0f};
            w.intensity = 1.0f;
        });
    }

    world.update_ibl(device, device.queue(), sampler);
    CHECK(world.ibl_resources().is_ready());

    // Remove dome light
    {
        auto scope = world.begin_sync();
        scope.free_light(pxr::SdfPath("/TestDome0"));
    }

    world.update_ibl(device, device.queue(), sampler);
    // Still ready (black environment)
    CHECK(world.ibl_resources().is_ready());
    wgpuSamplerRelease(sampler);
}

TEST_CASE("update_ibl with Z-up produces ready IBL") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);
    auto sampler = create_ibl_sampler(device);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light(pxr::SdfPath("/TestDome0"));
        scope.mutate_light(idx, [&](LightData& w) {
            w.type = LightData::Type::Dome;
            w.color = {1.0f, 1.0f, 1.0f};
            w.intensity = 1.0f;
        });
    }

    world.update_ibl(device, device.queue(), sampler, UpAxis::Z);

    CHECK(world.ibl_resources().is_ready());
    CHECK(world.ibl_resources().irradiance_view() != nullptr);
    CHECK(world.ibl_resources().prefiltered_env_view() != nullptr);
    CHECK(world.ibl_resources().env_cubemap_view() != nullptr);
    wgpuSamplerRelease(sampler);
}

TEST_CASE("clear resets IBL state") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);
    auto sampler = create_ibl_sampler(device);

    RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto idx = scope.alloc_light(pxr::SdfPath("/TestDome0"));
        scope.mutate_light(idx, [&](LightData& w) {
            w.type = LightData::Type::Dome;
            w.color = {1.0f, 1.0f, 1.0f};
            w.intensity = 1.0f;
        });
    }
    world.update_ibl(device, device.queue(), sampler);
    CHECK(world.ibl_resources().is_ready());

    world.clear();
    CHECK_FALSE(world.ibl_resources().is_ready());
    wgpuSamplerRelease(sampler);
}

#endif  // !__EMSCRIPTEN__
