#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/renderWorld.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstring>

#ifndef __EMSCRIPTEN__

namespace {
auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("material_buf_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("material_buf_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}
}  // namespace

TEST_CASE("Material SSBO round-trip via storage buffer") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    // Prepare material data on CPU
    std::vector<pts::rendering::Material> materials(3);
    materials[0].diffuse_color = {0.8f, 0.2f, 0.1f};
    materials[0].metallic = 0.9f;
    materials[0].roughness = 0.3f;
    materials[0].opacity = 0.7f;

    materials[1].diffuse_color = {0.1f, 0.9f, 0.2f};
    materials[1].metallic = 0.0f;
    materials[1].roughness = 1.0f;
    materials[1].opacity = 1.0f;

    materials[2].diffuse_color = {0.2f, 0.4f, 0.9f};
    materials[2].metallic = 0.5f;
    materials[2].roughness = 0.5f;
    materials[2].opacity = 0.5f;

    auto buf_size = materials.size() * sizeof(pts::rendering::Material);

    // Create storage buffer with CopyDst | CopySrc so we can write then readback
    auto ssbo = device.create_buffer(
        buf_size, static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                                               WGPUBufferUsage_CopySrc));
    REQUIRE(ssbo.is_valid());
    CHECK(ssbo.size() == buf_size);

    // Upload material data to SSBO
    wgpuQueueWriteBuffer(device.queue(), ssbo.handle(), 0, materials.data(), buf_size);

    // Create a staging buffer for readback (MapRead | CopyDst)
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    staging_desc.size = buf_size;
    staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(device.handle(), &staging_desc);
    REQUIRE(staging);

    // Copy SSBO to staging buffer
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device.handle(), &enc_desc);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, ssbo.handle(), 0, staging, 0, buf_size);

    WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(device.queue(), 1, &cmd_buf);
    wgpuCommandBufferRelease(cmd_buf);
    wgpuCommandEncoderRelease(encoder);

    // Map the staging buffer for reading
    struct MapContext {
        bool done = false;
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
    };
    MapContext map_ctx;

    WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
    map_cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto* ctx = static_cast<MapContext*>(userdata1);
        ctx->status = status;
        ctx->done = true;
    };
    map_cb.userdata1 = &map_ctx;

    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, buf_size, map_cb);

    // Poll until map completes
    while (!map_ctx.done) {
        wgpuInstanceProcessEvents(device.instance());
    }
    REQUIRE(map_ctx.status == WGPUMapAsyncStatus_Success);

    // Read back and verify
    const void* mapped = wgpuBufferGetConstMappedRange(staging, 0, buf_size);
    REQUIRE(mapped);

    std::vector<pts::rendering::Material> readback(3);
    std::memcpy(readback.data(), mapped, buf_size);
    wgpuBufferUnmap(staging);

    for (size_t i = 0; i < materials.size(); ++i) {
        CHECK(readback[i].diffuse_color.x == doctest::Approx(materials[i].diffuse_color.x));
        CHECK(readback[i].diffuse_color.y == doctest::Approx(materials[i].diffuse_color.y));
        CHECK(readback[i].diffuse_color.z == doctest::Approx(materials[i].diffuse_color.z));
        CHECK(readback[i].metallic == doctest::Approx(materials[i].metallic));
        CHECK(readback[i].roughness == doctest::Approx(materials[i].roughness));
        CHECK(readback[i].opacity == doctest::Approx(materials[i].opacity));
    }

    wgpuBufferRelease(staging);
}

TEST_CASE("Empty material buffer has minimum size for bind group validity") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    // Even with zero materials, the SSBO must be at least 32 bytes for a valid bind group
    constexpr uint32_t k_min_material_buffer_size = 32;
    auto ssbo = device.create_buffer(
        k_min_material_buffer_size,
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));

    REQUIRE(ssbo.is_valid());
    CHECK(ssbo.size() >= k_min_material_buffer_size);

    // Verify a bind group layout with read-only storage can be created
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Fragment;
    entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entry.buffer.minBindingSize = 0;

    WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_desc.entryCount = 1;
    bgl_desc.entries = &entry;
    auto layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);
    REQUIRE(layout);

    // Verify a bind group can be created with the minimum-size buffer
    WGPUBindGroupEntry bg_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bg_entry.binding = 0;
    bg_entry.buffer = ssbo.handle();
    bg_entry.offset = 0;
    bg_entry.size = k_min_material_buffer_size;

    WGPUBindGroupDescriptor bg_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bg_desc.layout = layout;
    bg_desc.entryCount = 1;
    bg_desc.entries = &bg_entry;
    auto bind_group = wgpuDeviceCreateBindGroup(device.handle(), &bg_desc);
    REQUIRE(bind_group);

    wgpuBindGroupRelease(bind_group);
    wgpuBindGroupLayoutRelease(layout);
}

#endif  // !__EMSCRIPTEN__
