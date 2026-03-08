#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/frameGraph.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace {

auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("frame_graph_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("frame_graph_test");
    }
    logger->set_level(spdlog::level::debug);
    return logger;
}

struct TestFixture {
    std::shared_ptr<spdlog::logger> logger = create_test_logger();
    pts::webgpu::Device device = pts::webgpu::Device::create(logger);
    pts::rendering::FrameGraph graph{device, logger};

    void submit(WGPUCommandEncoder encoder) {
        WGPUCommandBufferDescriptor cmd_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        auto cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
        wgpuQueueSubmit(device.queue(), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }

    auto create_encoder() -> WGPUCommandEncoder {
        WGPUCommandEncoderDescriptor enc_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        return wgpuDeviceCreateCommandEncoder(device.handle(), &enc_desc);
    }
};

}  // namespace

TEST_CASE("FrameGraph - single-pass Clear") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto color = f.graph.create("color", desc);
    f.graph.add_pass("clear_pass").color(color).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    CHECK(f.graph.get_texture_ref(color).view() != nullptr);

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - two-pass Clear then Load") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;

    auto color = f.graph.create("color", desc);
    f.graph.add_pass("first_pass").color(color).execute([](WGPURenderPassEncoder) {});

    f.graph.add_pass("second_pass").color(color).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - depth read-only") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc color_desc;
    color_desc.width = 64;
    color_desc.height = 64;
    color_desc.format = WGPUTextureFormat_BGRA8Unorm;

    pts::rendering::TextureDesc depth_desc;
    depth_desc.width = 64;
    depth_desc.height = 64;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto color0 = f.graph.create("color0", color_desc);
    auto depth = f.graph.create("depth", depth_desc);
    auto color1 = f.graph.create("color1", color_desc);

    f.graph.add_pass("depth_write_pass")
        .color(color0)
        .depth(depth)
        .execute([](WGPURenderPassEncoder) {});

    f.graph.add_pass("depth_read_pass")
        .color(color1)
        .depth_readonly(depth)
        .execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - backward dependency throws") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;

    pts::rendering::TextureDesc depth_desc;
    depth_desc.width = 64;
    depth_desc.height = 64;
    depth_desc.format = WGPUTextureFormat_Depth24Plus;

    auto depth_res = f.graph.create("depth", depth_desc);

    // Pass 0 reads depth that won't be written until pass 1
    f.graph.add_pass("reader")
        .color(f.graph.create("color0", desc))
        .depth_readonly(depth_res)
        .execute([](WGPURenderPassEncoder) {});

    f.graph.add_pass("writer")
        .color(f.graph.create("color1", desc))
        .depth(depth_res)
        .execute([](WGPURenderPassEncoder) {});

    CHECK_THROWS_AS(f.graph.compile(), std::runtime_error);
}

TEST_CASE("FrameGraph - cache reuse on same desc") {
    TestFixture f;

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    // Frame 1
    f.graph.begin_frame();
    auto h1 = f.graph.create("color", desc);
    f.graph.add_pass("pass").color(h1).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view1 = f.graph.get_texture_ref(h1).view();

    // Frame 2 - same desc, should reuse
    f.graph.begin_frame();
    auto h2 = f.graph.create("color", desc);
    f.graph.add_pass("pass").color(h2).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view2 = f.graph.get_texture_ref(h2).view();

    CHECK(view1 == view2);
}

TEST_CASE("FrameGraph - cache invalidation on resize") {
    TestFixture f;

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    // Frame 1
    f.graph.begin_frame();
    auto h1 = f.graph.create("color", desc);
    f.graph.add_pass("pass").color(h1).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view1 = f.graph.get_texture_ref(h1).view();

    // Frame 2 - different size
    desc.width = 128;
    desc.height = 128;

    f.graph.begin_frame();
    auto h2 = f.graph.create("color", desc);
    f.graph.add_pass("pass").color(h2).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view2 = f.graph.get_texture_ref(h2).view();

    CHECK(view1 != nullptr);
    CHECK(view2 != nullptr);
    // Note: cannot compare view1 != view2 — Dawn may reuse pointers after destruction.
    // The key invariant is that compile() succeeds with the new desc and produces a valid view.
}

TEST_CASE("FrameGraph - cache eviction of unused resources") {
    TestFixture f;

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    // Frame 1 - create "color_a" and "color_b"
    f.graph.begin_frame();
    auto a1 = f.graph.create("color_a", desc);
    auto b1 = f.graph.create("color_b", desc);
    f.graph.add_pass("pass_a").color(a1).execute([](WGPURenderPassEncoder) {});
    f.graph.add_pass("pass_b").color(b1).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();

    // Frame 2 - only "color_a", "color_b" should be evicted
    f.graph.begin_frame();
    auto ha = f.graph.create("color_a", desc);
    f.graph.add_pass("pass_a").color(ha).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();

    // color_a should still exist
    CHECK(f.graph.get_texture_ref(ha).view() != nullptr);
}

TEST_CASE("FrameGraph - TextureRef survives cache invalidation") {
    TestFixture f;

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    // Frame 1 — create texture and hold a TextureRef
    f.graph.begin_frame();
    auto h1 = f.graph.create("color", desc);
    f.graph.add_pass("pass").color(h1).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto ref1 = f.graph.get_texture_ref(h1);
    CHECK(ref1.view() != nullptr);

    // Frame 2 — resize triggers cache invalidation
    desc.width = 128;
    desc.height = 128;

    f.graph.begin_frame();
    auto h2 = f.graph.create("color", desc);
    f.graph.add_pass("pass").color(h2).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto ref2 = f.graph.get_texture_ref(h2);

    // Old ref still holds a valid (non-null) view via ref-counting
    CHECK(ref1.view() != nullptr);
    CHECK(ref2.view() != nullptr);
    CHECK(ref1.view() != ref2.view());
}

TEST_CASE("FrameGraph - read() backward dependency throws") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;

    auto color = f.graph.create("color", desc);

    // Pass 0 reads color that won't be written until pass 1
    f.graph.add_pass("reader")
        .color(f.graph.create("surface", desc))
        .read(color)
        .execute([](WGPURenderPassEncoder) {});

    f.graph.add_pass("writer").color(color).execute([](WGPURenderPassEncoder) {});

    CHECK_THROWS_AS(f.graph.compile(), std::runtime_error);
}

TEST_CASE("FrameGraph - read() valid forward dependency") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;

    auto color = f.graph.create("color", desc);

    // Pass 0 writes color, pass 1 reads it — valid forward dependency
    f.graph.add_pass("writer").color(color).execute([](WGPURenderPassEncoder) {});

    f.graph.add_pass("reader")
        .color(f.graph.create("surface", desc))
        .read(color)
        .execute([](WGPURenderPassEncoder) {});

    CHECK_NOTHROW(f.graph.compile());
}
