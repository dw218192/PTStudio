#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/profiling.h>
#include <core/rendering/iblResources.h>
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
    auto logger = spdlog::get("ibl_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("ibl_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

}  // namespace

#ifndef __EMSCRIPTEN__

TEST_CASE("IblPipelines init creates BRDF LUT and sampler") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    IblPipelines pipes;
    pipes.init(device, device.queue());

    CHECK(pipes.is_ready());
    CHECK(pipes.brdf_lut_view() != nullptr);
    CHECK(pipes.sampler() != nullptr);
}

TEST_CASE("IblResources set_uniform_environment transitions to ready") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    IblPipelines pipes;
    pipes.init(device, device.queue());

    IblResources ibl;
    ibl.set_uniform_environment(device, device.queue(), 0.5f, 0.5f, 0.5f);

    CHECK(ibl.is_ready());
    CHECK(ibl.prefiltered_env_view() != nullptr);
    CHECK(ibl.env_cubemap_view() != nullptr);
    CHECK(ibl.irradiance_view() != nullptr);
}

TEST_CASE("IblResources set_environment with synthetic HDR data") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    IblPipelines pipes;
    pipes.init(device, device.queue());

    IblResources ibl;

    constexpr uint32_t w = 16;
    constexpr uint32_t h = 8;
    std::vector<float> hdr(w * h * 4, 0.0f);
    for (size_t i = 0; i < w * h; ++i) {
        hdr[i * 4 + 0] = 1.0f;
        hdr[i * 4 + 1] = 0.5f;
        hdr[i * 4 + 2] = 0.25f;
        hdr[i * 4 + 3] = 1.0f;
    }

    ibl.set_environment(pipes, device, device.queue(), hdr.data(), w, h);

    CHECK(ibl.is_ready());
    CHECK(ibl.prefiltered_env_view() != nullptr);
    CHECK(ibl.env_cubemap_view() != nullptr);
    CHECK(ibl.irradiance_view() != nullptr);
    CHECK(ibl.prefiltered_env_view() != ibl.env_cubemap_view());
}

TEST_CASE("IblResources set_uniform_environment can be called again") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    IblResources ibl;
    ibl.set_uniform_environment(device, device.queue(), 1.0f, 0.0f, 0.0f);
    CHECK(ibl.is_ready());

    ibl.set_uniform_environment(device, device.queue(), 0.0f, 0.0f, 1.0f);
    CHECK(ibl.is_ready());
    CHECK(ibl.prefiltered_env_view() != nullptr);
    CHECK(ibl.env_cubemap_view() != nullptr);
    CHECK(ibl.irradiance_view() != nullptr);
}

TEST_CASE("IblResources move semantics") {
    auto logger = make_logger();
    auto device = pts::webgpu::Device::create(logger);

    IblResources ibl;
    ibl.set_uniform_environment(device, device.queue(), 0.3f, 0.3f, 0.3f);
    CHECK(ibl.is_ready());

    IblResources moved = std::move(ibl);
    CHECK(moved.is_ready());
    CHECK(moved.prefiltered_env_view() != nullptr);
    CHECK(moved.env_cubemap_view() != nullptr);
    CHECK(moved.irradiance_view() != nullptr);
}

#endif
