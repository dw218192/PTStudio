#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/webgpu/bufferReadback.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace {
auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("readback_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("readback_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}
}  // namespace

TEST_CASE("BufferReadback - starts in IdleState") {
    pts::webgpu::BufferReadback rb;
    CHECK_FALSE(rb.is_pending());
}

TEST_CASE("BufferReadback - try_read_u32 returns nullopt when idle") {
    pts::webgpu::BufferReadback rb;
    CHECK_FALSE(rb.try_read_u32().has_value());
}

TEST_CASE("BufferReadback - full readback of known pixel value") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    // Create a 1x1 R32Uint texture with CopySrc so we can read it back
    WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    tex_desc.size = {1, 1, 1};
    tex_desc.format = WGPUTextureFormat_R32Uint;
    tex_desc.usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    tex_desc.dimension = WGPUTextureDimension_2D;
    tex_desc.mipLevelCount = 1;
    tex_desc.sampleCount = 1;
    WGPUTexture texture = wgpuDeviceCreateTexture(device.handle(), &tex_desc);
    REQUIRE(texture);

    // Write a known value (42) into the texture via queue write
    uint32_t pixel_value = 42;
    WGPUTexelCopyTextureInfo dst_tex = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst_tex.texture = texture;
    dst_tex.origin = {0, 0, 0};

    WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    layout.bytesPerRow = 256;
    layout.rowsPerImage = 1;

    WGPUExtent3D write_size = {1, 1, 1};
    wgpuQueueWriteTexture(device.queue(), &dst_tex, &pixel_value, sizeof(pixel_value), &layout,
                          &write_size);

    // Create encoder and issue readback
    WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device.handle(), &enc_desc);
    REQUIRE(encoder);

    pts::webgpu::BufferReadback rb;
    rb.request(encoder, texture, 0, 0, device.handle(), device.instance());
    CHECK(rb.is_pending());

    // Finish and submit the encoder
    WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    REQUIRE(cmd_buf);
    wgpuQueueSubmit(device.queue(), 1, &cmd_buf);
    wgpuCommandBufferRelease(cmd_buf);

    // Poll until mapped (tick_until_settled uses wgpuInstanceProcessEvents)
    rb.tick_until_settled();
    CHECK_FALSE(rb.is_pending());

    // Read the value
    auto result = rb.try_read_u32();
    REQUIRE(result.has_value());
    CHECK(result.value() == 42);

    // Should be back in idle
    CHECK_FALSE(rb.is_pending());
    CHECK_FALSE(rb.try_read_u32().has_value());

    // Cleanup
    wgpuTextureRelease(texture);
}
