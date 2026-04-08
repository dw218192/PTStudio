#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/halfFloat.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/pipelineBuilder.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "embedded_resources.h"

#ifndef __EMSCRIPTEN__

namespace {

auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("auto_exposure_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("auto_exposure_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

struct LuminanceParams {
    uint32_t width;
    uint32_t height;
    float adaptation_speed;
    float dt;
    uint32_t has_depth;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};
static_assert(sizeof(LuminanceParams) == 32);

struct ExposureResult {
    float auto_exposure;
    uint32_t frame_count;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(ExposureResult) == 16);

// Create a uniform-color RGBA16Float texture and return the raw handle.
auto create_uniform_hdr_texture(const pts::webgpu::Device& device, uint32_t w, uint32_t h, float r,
                                float g, float b) -> WGPUTexture {
    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.size = {w, h, 1};
    desc.format = WGPUTextureFormat_RGBA16Float;
    desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst);
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.dimension = WGPUTextureDimension_2D;
    auto tex = wgpuDeviceCreateTexture(device.handle(), &desc);

    std::vector<uint16_t> pixels(w * h * 4);
    uint16_t hr = pts::rendering::float_to_half(r), hg = pts::rendering::float_to_half(g),
             hb = pts::rendering::float_to_half(b), ha = pts::rendering::float_to_half(1.0f);
    for (uint32_t i = 0; i < w * h; ++i) {
        pixels[i * 4 + 0] = hr;
        pixels[i * 4 + 1] = hg;
        pixels[i * 4 + 2] = hb;
        pixels[i * 4 + 3] = ha;
    }

    WGPUTexelCopyBufferLayout layout = {};
    layout.bytesPerRow = w * 4 * sizeof(uint16_t);
    layout.rowsPerImage = h;
    WGPUTexelCopyTextureInfo dest = {};
    dest.texture = tex;
    dest.aspect = WGPUTextureAspect_All;
    WGPUExtent3D extent = {w, h, 1};
    wgpuQueueWriteTexture(device.queue(), &dest, pixels.data(), pixels.size() * sizeof(uint16_t),
                          &layout, &extent);
    return tex;
}

// Map a buffer and read back its contents synchronously.
auto readback_buffer(const pts::webgpu::Device& device, WGPUBuffer src, uint64_t size)
    -> std::vector<uint8_t> {
    // Create staging buffer
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.size = size;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    auto staging = wgpuDeviceCreateBuffer(device.handle(), &staging_desc);

    // Copy src → staging
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    auto encoder = wgpuDeviceCreateCommandEncoder(device.handle(), &enc_desc);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, staging, 0, size);
    WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    auto cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(device.queue(), 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    // Map
    struct MapCtx {
        bool done = false;
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    } ctx;

    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
        auto* c = static_cast<MapCtx*>(ud1);
        c->status = status;
        c->done = true;
    };
    cb.userdata1 = &ctx;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, cb);
    while (!ctx.done) {
        wgpuInstanceProcessEvents(device.instance());
    }

    std::vector<uint8_t> result(static_cast<size_t>(size));
    if (ctx.status == WGPUMapAsyncStatus_Success) {
        const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, size);
        std::memcpy(result.data(), mapped, static_cast<size_t>(size));
        wgpuBufferUnmap(staging);
    }
    wgpuBufferRelease(staging);
    return result;
}

struct ComputeFixture {
    std::shared_ptr<spdlog::logger> logger = create_test_logger();
    pts::webgpu::Device device = pts::webgpu::Device::create(logger);
    pts::webgpu::ShaderModule shader{[&] {
        auto src = editor_resources::get_resource("editor/generated/shaders/luminance.wgsl");
        return device.create_shader_module_from_source(*src);
    }()};

    WGPUBindGroupLayout bgl = nullptr;
    WGPUPipelineLayout pl = nullptr;
    pts::webgpu::ComputePipeline pipeline{[&] {
        WGPUBindGroupLayoutEntry entries[5] = {};

        entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Compute;
        entries[0].texture.sampleType = WGPUTextureSampleType_Float;
        entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Compute;
        entries[1].sampler.type = WGPUSamplerBindingType_Filtering;

        entries[2] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Compute;
        entries[2].buffer.type = WGPUBufferBindingType_Storage;
        entries[2].buffer.minBindingSize = sizeof(ExposureResult);

        entries[3] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Compute;
        entries[3].buffer.type = WGPUBufferBindingType_Uniform;
        entries[3].buffer.minBindingSize = sizeof(LuminanceParams);

        entries[4] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[4].binding = 4;
        entries[4].visibility = WGPUShaderStage_Compute;
        entries[4].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[4].texture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bgl_desc.entryCount = 5;
        bgl_desc.entries = entries;
        bgl = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);

        WGPUPipelineLayoutDescriptor pl_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pl_desc.bindGroupLayoutCount = 1;
        pl_desc.bindGroupLayouts = &bgl;
        pl = wgpuDeviceCreatePipelineLayout(device.handle(), &pl_desc);

        return pts::webgpu::ComputePipelineBuilder(device)
            .shader(shader)
            .entry_point("cs_main")
            .pipeline_layout(pl)
            .build();
    }()};

    WGPUSampler sampler = [&] {
        WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
        desc.magFilter = WGPUFilterMode_Linear;
        desc.minFilter = WGPUFilterMode_Linear;
        desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        return wgpuDeviceCreateSampler(device.handle(), &desc);
    }();

    ~ComputeFixture() {
        if (pl) wgpuPipelineLayoutRelease(pl);
        if (bgl) wgpuBindGroupLayoutRelease(bgl);
        if (sampler) wgpuSamplerRelease(sampler);
    }

    // Run the luminance compute pass and return the ExposureResult.
    auto run(WGPUTexture hdr_tex, uint32_t w, uint32_t h, float adaptation_speed, float dt,
             WGPUBuffer result_buf) -> ExposureResult {
        WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        view_desc.format = WGPUTextureFormat_RGBA16Float;
        view_desc.dimension = WGPUTextureViewDimension_2D;
        view_desc.mipLevelCount = 1;
        view_desc.arrayLayerCount = 1;
        auto view = wgpuTextureCreateView(hdr_tex, &view_desc);

        // Create params uniform buffer
        WGPUBufferDescriptor params_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        params_desc.size = sizeof(LuminanceParams);
        params_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        auto params_buf = wgpuDeviceCreateBuffer(device.handle(), &params_desc);

        LuminanceParams params{w, h, adaptation_speed, dt, 0, 0, 0, 0};
        wgpuQueueWriteBuffer(device.queue(), params_buf, 0, &params, sizeof(params));

        // Dummy depth texture (has_depth=0, so shader won't read it)
        WGPUTextureDescriptor depth_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        depth_desc.size = {1, 1, 1};
        depth_desc.format = WGPUTextureFormat_Depth32Float;
        depth_desc.usage = WGPUTextureUsage_TextureBinding;
        depth_desc.mipLevelCount = 1;
        depth_desc.dimension = WGPUTextureDimension_2D;
        auto depth_tex = wgpuDeviceCreateTexture(device.handle(), &depth_desc);
        WGPUTextureViewDescriptor depth_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        depth_view_desc.format = WGPUTextureFormat_Depth32Float;
        depth_view_desc.dimension = WGPUTextureViewDimension_2D;
        depth_view_desc.mipLevelCount = 1;
        depth_view_desc.arrayLayerCount = 1;
        auto depth_view = wgpuTextureCreateView(depth_tex, &depth_view_desc);

        // Create bind group
        WGPUBindGroupEntry bg_entries[5] = {};
        bg_entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entries[0].binding = 0;
        bg_entries[0].textureView = view;
        bg_entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entries[1].binding = 1;
        bg_entries[1].sampler = sampler;
        bg_entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entries[2].binding = 2;
        bg_entries[2].buffer = result_buf;
        bg_entries[2].size = sizeof(ExposureResult);
        bg_entries[3] = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entries[3].binding = 3;
        bg_entries[3].buffer = params_buf;
        bg_entries[3].size = sizeof(LuminanceParams);
        bg_entries[4] = WGPU_BIND_GROUP_ENTRY_INIT;
        bg_entries[4].binding = 4;
        bg_entries[4].textureView = depth_view;

        WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bg_desc.layout = bgl;
        bg_desc.entryCount = 5;
        bg_desc.entries = bg_entries;
        auto bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);

        // Dispatch
        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        auto encoder = wgpuDeviceCreateCommandEncoder(device.handle(), &enc_desc);
        WGPUComputePassDescriptor cp_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        auto pass = wgpuCommandEncoderBeginComputePass(encoder, &cp_desc);
        wgpuComputePassEncoderSetPipeline(pass, pipeline.handle());
        wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        auto cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        wgpuQueueSubmit(device.queue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);

        // Read back
        auto data = readback_buffer(device, result_buf, sizeof(ExposureResult));
        ExposureResult out{};
        std::memcpy(&out, data.data(), sizeof(out));

        wgpuBindGroupRelease(bind_group);
        wgpuBufferRelease(params_buf);
        wgpuTextureViewRelease(depth_view);
        wgpuTextureRelease(depth_tex);
        wgpuTextureViewRelease(view);
        return out;
    }
};

}  // namespace

TEST_CASE("Auto-exposure: middle gray produces near-zero exposure correction") {
    ComputeFixture fx;

    // Middle gray (0.18) — auto-exposure should compute ~0.0 EV correction
    constexpr uint32_t w = 16, h = 16;
    auto tex = create_uniform_hdr_texture(fx.device, w, h, 0.18f, 0.18f, 0.18f);

    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = sizeof(ExposureResult);
    buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    auto result_buf = wgpuDeviceCreateBuffer(fx.device.handle(), &buf_desc);

    // Zero-init result buffer
    ExposureResult zeros{};
    wgpuQueueWriteBuffer(fx.device.queue(), result_buf, 0, &zeros, sizeof(zeros));

    auto result = fx.run(tex, w, h, 2.0f, 1.0f / 60.0f, result_buf);

    CHECK(result.frame_count == 1);
    // Luminance of (0.18, 0.18, 0.18) ≈ 0.18
    // log2(0.18) ≈ -2.474, ev = -2.474 - log2(0.18) = 0, auto_exposure = 0
    CHECK(result.auto_exposure == doctest::Approx(0.0f).epsilon(0.15));

    wgpuBufferRelease(result_buf);
    wgpuTextureRelease(tex);
}

TEST_CASE("Auto-exposure: bright scene produces negative exposure correction") {
    ComputeFixture fx;

    // Bright scene (10.0 per channel) — should produce negative auto_exposure
    constexpr uint32_t w = 16, h = 16;
    auto tex = create_uniform_hdr_texture(fx.device, w, h, 10.0f, 10.0f, 10.0f);

    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = sizeof(ExposureResult);
    buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    auto result_buf = wgpuDeviceCreateBuffer(fx.device.handle(), &buf_desc);

    ExposureResult zeros{};
    wgpuQueueWriteBuffer(fx.device.queue(), result_buf, 0, &zeros, sizeof(zeros));

    auto result = fx.run(tex, w, h, 2.0f, 1.0f / 60.0f, result_buf);

    CHECK(result.frame_count == 1);
    // Luminance = 10.0, log2(10) ≈ 3.322
    // ev = 3.322 - log2(0.18) ≈ 3.322 + 2.474 ≈ 5.796
    // auto_exposure = -5.796
    CHECK(result.auto_exposure < -3.0f);
    CHECK(result.auto_exposure > -10.0f);

    wgpuBufferRelease(result_buf);
    wgpuTextureRelease(tex);
}

TEST_CASE("Auto-exposure: temporal smoothing blends toward target") {
    ComputeFixture fx;

    constexpr uint32_t w = 16, h = 16;
    auto tex = create_uniform_hdr_texture(fx.device, w, h, 10.0f, 10.0f, 10.0f);

    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = sizeof(ExposureResult);
    buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    auto result_buf = wgpuDeviceCreateBuffer(fx.device.handle(), &buf_desc);

    // First frame: snaps to target (frame_count == 0)
    ExposureResult zeros{};
    wgpuQueueWriteBuffer(fx.device.queue(), result_buf, 0, &zeros, sizeof(zeros));
    auto r1 = fx.run(tex, w, h, 2.0f, 1.0f / 60.0f, result_buf);
    CHECK(r1.frame_count == 1);
    float first_exposure = r1.auto_exposure;

    // Second frame: should be close to first (temporal smoothing from same scene)
    auto r2 = fx.run(tex, w, h, 2.0f, 1.0f / 60.0f, result_buf);
    CHECK(r2.frame_count == 2);
    // With same input, the target is the same, so smoothing moves toward same value
    CHECK(r2.auto_exposure == doctest::Approx(first_exposure).epsilon(0.5));

    wgpuBufferRelease(result_buf);
    wgpuTextureRelease(tex);
}

#else
// Emscripten: GPU compute tests not available in node harness
TEST_CASE("Auto-exposure: placeholder (emscripten)") {
    CHECK(true);
}
#endif
