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
    materials[0].emissive_color = {1.0f, 0.5f, 0.0f};
    materials[0].roughness = 0.3f;
    materials[0].opacity = 0.7f;
    materials[0].diffuse_tex = 42;

    materials[1].diffuse_color = {0.1f, 0.9f, 0.2f};
    materials[1].metallic = 0.0f;
    materials[1].emissive_color = {0.0f, 0.0f, 0.0f};
    materials[1].roughness = 1.0f;
    materials[1].opacity = 1.0f;

    materials[2].diffuse_color = {0.2f, 0.4f, 0.9f};
    materials[2].metallic = 0.5f;
    materials[2].emissive_color = {0.3f, 0.3f, 0.3f};
    materials[2].roughness = 0.5f;
    materials[2].opacity = 0.5f;
    materials[2].emissive_tex = 7;

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
        CHECK(readback[i].emissive_color.x == doctest::Approx(materials[i].emissive_color.x));
        CHECK(readback[i].emissive_color.y == doctest::Approx(materials[i].emissive_color.y));
        CHECK(readback[i].emissive_color.z == doctest::Approx(materials[i].emissive_color.z));
        CHECK(readback[i].roughness == doctest::Approx(materials[i].roughness));
        CHECK(readback[i].opacity == doctest::Approx(materials[i].opacity));
        CHECK(readback[i].diffuse_tex == materials[i].diffuse_tex);
        CHECK(readback[i].normal_tex == materials[i].normal_tex);
        CHECK(readback[i].metallic_tex == materials[i].metallic_tex);
        CHECK(readback[i].roughness_tex == materials[i].roughness_tex);
        CHECK(readback[i].emissive_tex == materials[i].emissive_tex);
        CHECK(readback[i].opacity_tex == materials[i].opacity_tex);
        CHECK(readback[i].tex_channels == materials[i].tex_channels);
    }

    wgpuBufferRelease(staging);
}

TEST_CASE("Empty material buffer has minimum size for bind group validity") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    // Even with zero materials, the SSBO must be at least sizeof(Material) for a valid bind group
    constexpr uint32_t k_min_material_buffer_size = sizeof(pts::rendering::Material);
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

TEST_CASE("prepare_gpu_buffers creates material buffer from world materials") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    pts::rendering::RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto& mats = scope.materials();
        pts::rendering::Material m{};
        m.diffuse_color = {0.5f, 0.6f, 0.7f};
        m.roughness = 0.8f;
        mats.push_back(m);
    }

    world.prepare_gpu_buffers(device, device.queue());

    CHECK(world.material_buffer().is_valid());
    CHECK(world.material_buffer().size() >= sizeof(pts::rendering::Material));
}

TEST_CASE("prepare_gpu_buffers creates light buffer with fallback when no lights") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    pts::rendering::RenderWorld world;
    // No lights added — should get fallback distant light
    {
        auto scope = world.begin_sync();
        // just bump versions
    }

    world.prepare_gpu_buffers(device, device.queue());

    CHECK(world.light_buffer().is_valid());
    CHECK(world.gpu_light_count() == 1);
}

TEST_CASE("prepare_gpu_buffers uploads active lights") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    pts::rendering::RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto l0 = scope.alloc_light_slot();
        {
            auto w = scope.write_light(l0);
            w->type = pts::rendering::LightData::Type::Distant;
            w->color = {1.0f, 0.0f, 0.0f};
            w->intensity = 2.0f;
        }

        auto l1 = scope.alloc_light_slot();
        {
            auto w = scope.write_light(l1);
            w->type = pts::rendering::LightData::Type::Sphere;
            w->color = {0.0f, 1.0f, 0.0f};
            w->intensity = 3.0f;
        }
    }

    world.prepare_gpu_buffers(device, device.queue());

    CHECK(world.light_buffer().is_valid());
    CHECK(world.gpu_light_count() == 2);
    CHECK(world.light_buffer().size() >= 2 * sizeof(pts::rendering::Light));
}

TEST_CASE("prepare_gpu_buffers skips upload when versions unchanged") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    pts::rendering::RenderWorld world;
    {
        auto scope = world.begin_sync();
        auto& mats = scope.materials();
        mats.push_back(pts::rendering::Material{});
    }

    world.prepare_gpu_buffers(device, device.queue());
    auto mat_buf_handle = world.material_buffer().handle();
    auto light_buf_handle = world.light_buffer().handle();

    // Call again without changes — buffers should be reused (same handle)
    world.prepare_gpu_buffers(device, device.queue());
    CHECK(world.material_buffer().handle() == mat_buf_handle);
    CHECK(world.light_buffer().handle() == light_buf_handle);
}

TEST_CASE("prepare_gpu_buffers creates placeholder texture array when no textures loaded") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    pts::rendering::RenderWorld world;
    {
        auto scope = world.begin_sync();
    }

    world.prepare_gpu_buffers(device, device.queue());

    CHECK(world.texture_array_view() != nullptr);
    CHECK(world.texture_sampler() != nullptr);
}

TEST_CASE("Material struct is 64 bytes") {
    CHECK(sizeof(pts::rendering::Material) == 64);
}

TEST_CASE("Material default texture indices are UINT32_MAX") {
    pts::rendering::Material mat{};
    CHECK(mat.diffuse_tex == UINT32_MAX);
    CHECK(mat.normal_tex == UINT32_MAX);
    CHECK(mat.metallic_tex == UINT32_MAX);
    CHECK(mat.roughness_tex == UINT32_MAX);
    CHECK(mat.emissive_tex == UINT32_MAX);
    CHECK(mat.opacity_tex == UINT32_MAX);
    CHECK(mat.tex_channels == 0);
}

TEST_CASE("clear resets GPU buffer state") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    pts::rendering::RenderWorld world;
    {
        auto scope = world.begin_sync();
        scope.materials().push_back(pts::rendering::Material{});
        auto l = scope.alloc_light_slot();
        {
            auto w = scope.write_light(l);
            w->type = pts::rendering::LightData::Type::Distant;
        }
    }

    world.prepare_gpu_buffers(device, device.queue());
    CHECK(world.material_buffer().is_valid());
    CHECK(world.light_buffer().is_valid());

    world.clear();
    CHECK_FALSE(world.material_buffer().is_valid());
    CHECK_FALSE(world.light_buffer().is_valid());
    CHECK(world.gpu_light_count() == 0);
    CHECK(world.texture_array_view() == nullptr);
    CHECK(world.texture_sampler() == nullptr);
}

#endif  // !__EMSCRIPTEN__
