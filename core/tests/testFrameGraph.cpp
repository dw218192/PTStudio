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

    pts::rendering::ResourceHandle color_handle;
    f.graph.add_pass("clear_pass", [&](pts::rendering::PassBuilder& builder) {
        color_handle = builder.create("color", desc);
        return [](WGPURenderPassEncoder) {};
    });

    f.graph.compile();

    auto view = f.graph.get_texture_view(color_handle);
    CHECK(view != nullptr);

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

    pts::rendering::ResourceHandle color_handle;
    f.graph.add_pass("first_pass", [&](pts::rendering::PassBuilder& builder) {
        color_handle = builder.create("color", desc);
        return [](WGPURenderPassEncoder) {};
    });

    f.graph.add_pass("second_pass", [&](pts::rendering::PassBuilder& builder) {
        builder.write_color(color_handle);
        return [](WGPURenderPassEncoder) {};
    });

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

    pts::rendering::ResourceHandle depth_handle;
    f.graph.add_pass("depth_write_pass", [&](pts::rendering::PassBuilder& builder) {
        builder.create("color0", color_desc);
        depth_handle = builder.create("depth", depth_desc);
        return [](WGPURenderPassEncoder) {};
    });

    f.graph.add_pass("depth_read_pass", [&](pts::rendering::PassBuilder& builder) {
        builder.create("color1", color_desc);
        builder.read_depth(depth_handle);
        return [](WGPURenderPassEncoder) {};
    });

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

    // Import a resource so both passes can reference it
    auto shared = f.graph.import("shared", nullptr, desc);

    // Pass 0 reads a resource that won't be written until pass 1
    f.graph.add_pass("reader", [&](pts::rendering::PassBuilder& builder) {
        builder.read_color(shared);
        return [](WGPURenderPassEncoder) {};
    });

    // Pass 1 is the first writer
    f.graph.add_pass("writer", [&](pts::rendering::PassBuilder& builder) {
        builder.write_color(shared);
        return [](WGPURenderPassEncoder) {};
    });

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
    pts::rendering::ResourceHandle h1;
    f.graph.add_pass("pass", [&](pts::rendering::PassBuilder& builder) {
        h1 = builder.create("color", desc);
        return [](WGPURenderPassEncoder) {};
    });
    f.graph.compile();
    auto view1 = f.graph.get_texture_view(h1);

    // Frame 2 - same desc, should reuse
    f.graph.begin_frame();
    pts::rendering::ResourceHandle h2;
    f.graph.add_pass("pass", [&](pts::rendering::PassBuilder& builder) {
        h2 = builder.create("color", desc);
        return [](WGPURenderPassEncoder) {};
    });
    f.graph.compile();
    auto view2 = f.graph.get_texture_view(h2);

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
    pts::rendering::ResourceHandle h1;
    f.graph.add_pass("pass", [&](pts::rendering::PassBuilder& builder) {
        h1 = builder.create("color", desc);
        return [](WGPURenderPassEncoder) {};
    });
    f.graph.compile();
    auto view1 = f.graph.get_texture_view(h1);

    // Frame 2 - different size
    desc.width = 128;
    desc.height = 128;

    f.graph.begin_frame();
    pts::rendering::ResourceHandle h2;
    f.graph.add_pass("pass", [&](pts::rendering::PassBuilder& builder) {
        h2 = builder.create("color", desc);
        return [](WGPURenderPassEncoder) {};
    });
    f.graph.compile();
    auto view2 = f.graph.get_texture_view(h2);

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
    f.graph.add_pass("pass_a", [&](pts::rendering::PassBuilder& builder) {
        builder.create("color_a", desc);
        return [](WGPURenderPassEncoder) {};
    });
    f.graph.add_pass("pass_b", [&](pts::rendering::PassBuilder& builder) {
        builder.create("color_b", desc);
        return [](WGPURenderPassEncoder) {};
    });
    f.graph.compile();

    // Frame 2 - only "color_a", "color_b" should be evicted
    f.graph.begin_frame();
    pts::rendering::ResourceHandle ha;
    f.graph.add_pass("pass_a", [&](pts::rendering::PassBuilder& builder) {
        ha = builder.create("color_a", desc);
        return [](WGPURenderPassEncoder) {};
    });
    f.graph.compile();

    // color_a should still exist
    CHECK(f.graph.get_texture_view(ha) != nullptr);
}
