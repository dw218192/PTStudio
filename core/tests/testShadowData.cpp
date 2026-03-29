#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/renderWorld.h>
#include <core/rendering/shadowMapPass.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

using namespace pts::rendering;

TEST_CASE("ShadowInfo struct is 80 bytes") {
    CHECK(sizeof(ShadowInfo) == 80);
}

TEST_CASE("ShadowInfo default has no shadow") {
    ShadowInfo info{};
    CHECK(info.has_shadow == 0);
    CHECK(info.layer == 0);
}

TEST_CASE("fresh world has no ShadowPassData") {
    RenderWorld world;
    CHECK(ShadowPassData::find(world) == nullptr);
}

#ifndef __EMSCRIPTEN__

namespace {
auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("shadow_data_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("shadow_data_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}
}  // namespace

TEST_CASE("upload creates valid buffer") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo active{};
    active.has_shadow = 1;
    active.layer = 0;
    active.texel_size = 1.0f / 1024.0f;
    active.normal_bias = 0.02f;

    ShadowInfo inactive{};  // has_shadow = 0

    std::vector<ShadowInfo> infos = {active, inactive};

    auto& sd = ShadowPassData::get_or_create(world);
    sd.upload(infos, device, device.queue());

    CHECK(sd.count == 1);  // only one active
    CHECK(sd.info_buffer.is_valid());
    CHECK(sd.info_buffer.size() >= 2 * sizeof(ShadowInfo));
}

TEST_CASE("upload grows buffer when capacity exceeded") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo info{};
    info.has_shadow = 1;
    std::vector<ShadowInfo> one_info = {info};

    auto& sd = ShadowPassData::get_or_create(world);
    sd.upload(one_info, device, device.queue());
    auto info_handle = sd.info_buffer.handle();

    // Grow to 3 entries
    std::vector<ShadowInfo> large(3, info);
    sd.upload(large, device, device.queue());

    CHECK(sd.count == 3);
    CHECK(sd.info_buffer.size() >= 3 * sizeof(ShadowInfo));
    CHECK(sd.info_buffer.handle() != info_handle);
}

TEST_CASE("upload reuses buffer when capacity sufficient") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo info{};
    info.has_shadow = 1;
    std::vector<ShadowInfo> infos(3, info);

    auto& sd = ShadowPassData::get_or_create(world);
    sd.upload(infos, device, device.queue());
    auto info_handle = sd.info_buffer.handle();

    // Fewer entries — buffer reused
    std::vector<ShadowInfo> fewer(1, info);
    sd.upload(fewer, device, device.queue());

    CHECK(sd.count == 1);
    CHECK(sd.info_buffer.handle() == info_handle);
}

TEST_CASE("upload with all inactive creates minimum-size buffer") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo inactive{};  // has_shadow = 0
    std::vector<ShadowInfo> infos = {inactive, inactive};

    auto& sd = ShadowPassData::get_or_create(world);
    sd.upload(infos, device, device.queue());

    CHECK(sd.count == 0);
    CHECK(sd.info_buffer.is_valid());
    CHECK(sd.info_buffer.size() >= 2 * sizeof(ShadowInfo));
}

#endif  // !__EMSCRIPTEN__
