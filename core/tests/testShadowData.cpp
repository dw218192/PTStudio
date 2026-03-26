#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/renderWorld.h>
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

TEST_CASE("clear resets shadow_count") {
    RenderWorld world;
    CHECK(world.shadow_count() == 0);
    world.clear();
    CHECK(world.shadow_count() == 0);
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

TEST_CASE("set_shadow_data creates valid buffer") {
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

    world.set_shadow_data(infos, device, device.queue());

    CHECK(world.shadow_count() == 1);  // only one active
    CHECK(world.shadow_info_buffer().is_valid());
    CHECK(world.shadow_info_buffer().size() >= 2 * sizeof(ShadowInfo));
}

TEST_CASE("set_shadow_data grows buffer when capacity exceeded") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo info{};
    info.has_shadow = 1;
    std::vector<ShadowInfo> one_info = {info};

    world.set_shadow_data(one_info, device, device.queue());
    auto info_handle = world.shadow_info_buffer().handle();

    // Grow to 3 entries
    std::vector<ShadowInfo> large(3, info);
    world.set_shadow_data(large, device, device.queue());

    CHECK(world.shadow_count() == 3);
    CHECK(world.shadow_info_buffer().size() >= 3 * sizeof(ShadowInfo));
    CHECK(world.shadow_info_buffer().handle() != info_handle);
}

TEST_CASE("set_shadow_data reuses buffer when capacity sufficient") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo info{};
    info.has_shadow = 1;
    std::vector<ShadowInfo> infos(3, info);

    world.set_shadow_data(infos, device, device.queue());
    auto info_handle = world.shadow_info_buffer().handle();

    // Fewer entries — buffer reused
    std::vector<ShadowInfo> fewer(1, info);
    world.set_shadow_data(fewer, device, device.queue());

    CHECK(world.shadow_count() == 1);
    CHECK(world.shadow_info_buffer().handle() == info_handle);
}

TEST_CASE("clear_shadow_data resets count but preserves buffer") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo info{};
    info.has_shadow = 1;
    std::vector<ShadowInfo> infos = {info};

    world.set_shadow_data(infos, device, device.queue());
    CHECK(world.shadow_count() == 1);

    world.clear_shadow_data();
    CHECK(world.shadow_count() == 0);
    CHECK(world.shadow_info_buffer().is_valid());
}

TEST_CASE("clear() resets shadow state") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo info{};
    info.has_shadow = 1;
    std::vector<ShadowInfo> infos = {info};

    world.set_shadow_data(infos, device, device.queue());
    world.clear();
    CHECK(world.shadow_count() == 0);
}

TEST_CASE("set_shadow_data with all inactive creates minimum-size buffer") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    RenderWorld world;

    ShadowInfo inactive{};  // has_shadow = 0
    std::vector<ShadowInfo> infos = {inactive, inactive};

    world.set_shadow_data(infos, device, device.queue());

    CHECK(world.shadow_count() == 0);
    CHECK(world.shadow_info_buffer().is_valid());
    CHECK(world.shadow_info_buffer().size() >= 2 * sizeof(ShadowInfo));
}

#endif  // !__EMSCRIPTEN__
