#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/device.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <thread>

#include "testApplication.h"

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
    depth_desc.format = WGPUTextureFormat_Depth32Float;

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
    depth_desc.format = WGPUTextureFormat_Depth32Float;

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

    CHECK(f.graph.cached_texture_count() == 2);

    // Frame 2 - only "color_a", "color_b" should be evicted
    f.graph.begin_frame();
    auto ha = f.graph.create("color_a", desc);
    f.graph.add_pass("pass_a").color(ha).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();

    // color_a should still exist, color_b should be evicted
    CHECK(f.graph.get_texture_ref(ha).view() != nullptr);
    CHECK(f.graph.cached_texture_count() == 1);
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

TEST_CASE("FrameGraph - MRT: two color attachments") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto color0 = f.graph.create("color0", desc);
    auto color1 = f.graph.create("color1", desc);

    bool executed = false;
    f.graph.add_pass("mrt_pass").color(color0).color(color1).execute([&](WGPURenderPassEncoder) {
        executed = true;
    });

    f.graph.compile();

    CHECK(f.graph.get_texture_ref(color0).view() != nullptr);
    CHECK(f.graph.get_texture_ref(color1).view() != nullptr);

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);

    CHECK(executed);
}

TEST_CASE("FrameGraph - MRT: second pass loads both attachments") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto color0 = f.graph.create("color0", desc);
    auto color1 = f.graph.create("color1", desc);

    // Pass 0 writes both attachments (first writer -> Clear)
    f.graph.add_pass("mrt_write").color(color0).color(color1).execute([](WGPURenderPassEncoder) {});

    // Pass 1 writes both again (not first writer -> Load)
    f.graph.add_pass("mrt_load").color(color0).color(color1).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - compute pass") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_RGBA8Unorm;

    auto storage_tex = f.graph.create("storage", desc);

    bool executed = false;
    f.graph.add_pass("compute_pass")
        .storage_write(storage_tex)
        .execute([&](WGPUComputePassEncoder) { executed = true; });

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);

    CHECK(executed);
}

TEST_CASE("FrameGraph - compute then render pass") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc storage_desc;
    storage_desc.width = 64;
    storage_desc.height = 64;
    storage_desc.format = WGPUTextureFormat_RGBA8Unorm;

    auto storage_tex = f.graph.create("storage", storage_desc);

    // Compute pass writes storage texture
    f.graph.add_pass("compute").storage_write(storage_tex).execute([](WGPUComputePassEncoder) {});

    pts::rendering::TextureDesc color_desc;
    color_desc.width = 64;
    color_desc.height = 64;
    color_desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto color = f.graph.create("color", color_desc);

    // Render pass reads storage texture result
    f.graph.add_pass("render").color(color).read(storage_tex).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - find_or_create creates on first call") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto h1 = f.graph.find_or_create("color", desc);
    CHECK(h1.is_valid());
    CHECK(h1.index == 0);
}

TEST_CASE("FrameGraph - find_or_create returns existing handle") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto h1 = f.graph.find_or_create("color", desc);
    auto h2 = f.graph.find_or_create("color", desc);

    CHECK(h1.index == h2.index);
}

TEST_CASE("FrameGraph - find_or_create different names create different handles") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto h1 = f.graph.find_or_create("color_a", desc);
    auto h2 = f.graph.find_or_create("color_b", desc);

    CHECK(h1.index != h2.index);
}

TEST_CASE("FrameGraph - find returns nullopt for missing resource") {
    TestFixture f;

    f.graph.begin_frame();

    auto result = f.graph.find("nonexistent");
    CHECK(!result.has_value());
}

TEST_CASE("FrameGraph - find returns handle for existing resource") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto h1 = f.graph.find_or_create("color", desc);
    auto found = f.graph.find("color");

    REQUIRE(found.has_value());
    CHECK(found->index == h1.index);
}

TEST_CASE("FrameGraph - picking texture CopySrc readback") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc color_desc;
    color_desc.width = 64;
    color_desc.height = 64;
    color_desc.format = WGPUTextureFormat_BGRA8Unorm;

    pts::rendering::TextureDesc picking_desc;
    picking_desc.width = 64;
    picking_desc.height = 64;
    picking_desc.format = WGPUTextureFormat_R32Uint;
    picking_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    picking_desc.clear_color = {static_cast<double>(UINT32_MAX), 0, 0, 0};

    auto color = f.graph.create("scene_color", color_desc);
    auto picking = f.graph.create("picking_ids", picking_desc);

    f.graph.add_pass("forward").color(color).color(picking).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    // Verify both textures were allocated
    auto color_ref = f.graph.get_texture_ref(color);
    auto picking_ref = f.graph.get_texture_ref(picking);
    CHECK(color_ref.view() != nullptr);
    CHECK(picking_ref.view() != nullptr);
    CHECK(picking_ref.texture() != nullptr);

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);

    // Create a readback buffer (256 bytes = WebGPU minimum bytesPerRow)
    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    auto readback = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    REQUIRE(readback != nullptr);

    // Copy a single pixel from the picking texture to the readback buffer
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = picking_ref.texture();
    src.mipLevel = 0;
    src.origin = {0, 0, 0};

    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = readback;
    dst.layout.offset = 0;
    dst.layout.bytesPerRow = 256;
    dst.layout.rowsPerImage = 1;

    WGPUExtent3D extent = {1, 1, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    f.submit(encoder);

    // Map and read back — the clear color should be UINT32_MAX (sentinel)
    WGPUBufferMapCallbackInfo map_cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    map_cb.mode = WGPUCallbackMode_AllowProcessEvents;
    map_cb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void*, void*) {};
    wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, 256, map_cb);

    // Poll until the GPU work completes and the buffer is mapped
    while (wgpuBufferGetMapState(readback) != WGPUBufferMapState_Mapped) {
        wgpuInstanceProcessEvents(f.device.instance());
        std::this_thread::yield();
    }

    REQUIRE(wgpuBufferGetMapState(readback) == WGPUBufferMapState_Mapped);
    auto* data =
        static_cast<const uint32_t*>(wgpuBufferGetConstMappedRange(readback, 0, sizeof(uint32_t)));
    REQUIRE(data != nullptr);
    CHECK(*data == UINT32_MAX);
    wgpuBufferUnmap(readback);

    wgpuBufferRelease(readback);
}

TEST_CASE("FrameGraph - usage auto-inference from read()") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;
    // Start with only RenderAttachment
    desc.usage = WGPUTextureUsage_RenderAttachment;

    auto color = f.graph.create("color", desc);

    // Pass 0 writes color
    f.graph.add_pass("writer").color(color).execute([](WGPURenderPassEncoder) {});

    pts::rendering::TextureDesc surface_desc;
    surface_desc.width = 64;
    surface_desc.height = 64;
    surface_desc.format = WGPUTextureFormat_BGRA8Unorm;

    // Pass 1 reads color — should auto-add TextureBinding
    f.graph.add_pass("reader")
        .color(f.graph.create("surface", surface_desc))
        .read(color)
        .execute([](WGPURenderPassEncoder) {});

    // Should compile and execute without error (TextureBinding auto-inferred)
    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

// --- Buffer tests ---

TEST_CASE("FrameGraph - create buffer, verify handle and ref") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::BufferDesc desc;
    desc.size = 1024;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    auto h = f.graph.find_or_create_buffer("my_buffer", desc);
    CHECK(h.is_valid());

    f.graph.compile();

    auto ref = f.graph.get_buffer_ref(h);
    CHECK(static_cast<bool>(ref));
    CHECK(ref.handle() != nullptr);
    CHECK(ref.size() == 1024);
}

TEST_CASE("FrameGraph - find_or_create_buffer returns same handle on second call") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::BufferDesc desc;
    desc.size = 1024;
    desc.usage = WGPUBufferUsage_Storage;

    auto h1 = f.graph.find_or_create_buffer("buf", desc);
    auto h2 = f.graph.find_or_create_buffer("buf", desc);
    CHECK(h1.index == h2.index);
}

TEST_CASE("FrameGraph - find_or_create_buffer larger size triggers realloc + version bump") {
    TestFixture f;

    pts::rendering::BufferDesc desc;
    desc.size = 512;
    desc.usage = WGPUBufferUsage_Storage;

    // Frame 1 — small buffer
    f.graph.begin_frame();
    auto h1 = f.graph.find_or_create_buffer("buf", desc);
    f.graph.compile();
    auto ref1 = f.graph.get_buffer_ref(h1);
    CHECK(ref1.handle() != nullptr);
    auto v1 = ref1.size();
    CHECK(v1 == 512);

    // Frame 2 — larger size triggers reallocation
    desc.size = 2048;
    f.graph.begin_frame();
    auto h2 = f.graph.find_or_create_buffer("buf", desc);
    f.graph.compile();
    auto ref2 = f.graph.get_buffer_ref(h2);
    CHECK(ref2.handle() != nullptr);
    CHECK(ref2.size() == 2048);
}

TEST_CASE("FrameGraph - import_buffer same pointer reuses (same version)") {
    TestFixture f;

    // Create an external buffer to import
    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform;
    auto ext_buf = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    REQUIRE(ext_buf != nullptr);

    // Frame 1 — import
    f.graph.begin_frame();
    auto h1 = f.graph.import_buffer("imported", ext_buf, 256);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 1);

    // Frame 2 — same pointer, should reuse
    f.graph.begin_frame();
    auto h2 = f.graph.import_buffer("imported", ext_buf, 256);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 1);
    auto ref = f.graph.get_buffer_ref(h2);
    CHECK(ref.handle() == ext_buf);

    wgpuBufferDestroy(ext_buf);
    wgpuBufferRelease(ext_buf);
}

TEST_CASE("FrameGraph - import_buffer different pointer bumps version") {
    TestFixture f;

    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform;
    auto ext_buf1 = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    auto ext_buf2 = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    REQUIRE(ext_buf1 != nullptr);
    REQUIRE(ext_buf2 != nullptr);

    // Frame 1 — import buf1
    f.graph.begin_frame();
    f.graph.import_buffer("imported", ext_buf1, 256);
    f.graph.compile();

    // Frame 2 — import buf2 (different pointer)
    f.graph.begin_frame();
    auto h2 = f.graph.import_buffer("imported", ext_buf2, 256);
    f.graph.compile();
    auto ref = f.graph.get_buffer_ref(h2);
    CHECK(ref.handle() == ext_buf2);

    wgpuBufferDestroy(ext_buf1);
    wgpuBufferRelease(ext_buf1);
    wgpuBufferDestroy(ext_buf2);
    wgpuBufferRelease(ext_buf2);
}

TEST_CASE("FrameGraph - buffer eviction when not used next frame") {
    TestFixture f;

    pts::rendering::BufferDesc desc;
    desc.size = 512;
    desc.usage = WGPUBufferUsage_Storage;

    // Frame 1 — create buffer
    f.graph.begin_frame();
    f.graph.find_or_create_buffer("buf_a", desc);
    f.graph.find_or_create_buffer("buf_b", desc);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 2);

    // Frame 2 — only use buf_a, buf_b should be evicted
    f.graph.begin_frame();
    f.graph.find_or_create_buffer("buf_a", desc);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 1);
}

TEST_CASE("FrameGraph - find_buffer") {
    TestFixture f;

    f.graph.begin_frame();

    CHECK(!f.graph.find_buffer("nonexistent").has_value());

    pts::rendering::BufferDesc desc;
    desc.size = 128;
    desc.usage = WGPUBufferUsage_Uniform;

    auto h = f.graph.find_or_create_buffer("my_buf", desc);
    auto found = f.graph.find_buffer("my_buf");
    REQUIRE(found.has_value());
    CHECK(found->index == h.index);
}

TEST_CASE("FrameGraph - cached_buffer_count") {
    TestFixture f;

    f.graph.begin_frame();
    CHECK(f.graph.cached_buffer_count() == 0);

    pts::rendering::BufferDesc desc;
    desc.size = 64;
    desc.usage = WGPUBufferUsage_Storage;

    f.graph.find_or_create_buffer("a", desc);
    f.graph.find_or_create_buffer("b", desc);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 2);
}

// --- Array texture tests ---

TEST_CASE("FrameGraph - array texture creates N+1 views") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.array_layers = 4;
    desc.format = WGPUTextureFormat_Depth32Float;

    auto h = f.graph.create("shadow_array", desc);
    f.graph.add_pass("shadow0").depth(h, 0).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    auto ref = f.graph.get_texture_ref(h);
    CHECK(ref.view() != nullptr);
    CHECK(ref.layer_count() == 4);
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(ref.layer_view(i) != nullptr);
    }
}

TEST_CASE("FrameGraph - layer_view returns distinct per-layer views") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.array_layers = 4;
    desc.format = WGPUTextureFormat_Depth32Float;

    auto h = f.graph.create("shadow_array", desc);
    f.graph.add_pass("shadow0").depth(h, 0).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    auto ref = f.graph.get_texture_ref(h);
    // Each layer view should be distinct from the array view and from each other
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(ref.layer_view(i) != ref.view());
        for (uint32_t j = i + 1; j < 4; ++j) {
            CHECK(ref.layer_view(i) != ref.layer_view(j));
        }
    }
}

TEST_CASE("FrameGraph - descs_match returns false when array_layers differs") {
    TestFixture f;

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_Depth32Float;
    desc.array_layers = 4;

    // Frame 1 — create with 4 layers
    f.graph.begin_frame();
    auto h1 = f.graph.create("shadow", desc);
    f.graph.add_pass("pass").depth(h1, 0).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto ref1 = f.graph.get_texture_ref(h1);
    CHECK(ref1.layer_count() == 4);

    // Frame 2 — change to 2 layers, should NOT reuse the cached texture
    desc.array_layers = 2;
    f.graph.begin_frame();
    auto h2 = f.graph.create("shadow", desc);
    f.graph.add_pass("pass").depth(h2, 0).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto ref2 = f.graph.get_texture_ref(h2);
    CHECK(ref2.layer_count() == 2);
}

TEST_CASE("FrameGraph - depth attachment with layer index executes") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc color_desc;
    color_desc.width = 64;
    color_desc.height = 64;
    color_desc.format = WGPUTextureFormat_BGRA8Unorm;

    pts::rendering::TextureDesc depth_desc;
    depth_desc.width = 64;
    depth_desc.height = 64;
    depth_desc.array_layers = 4;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto color = f.graph.create("color", color_desc);
    auto depth = f.graph.create("shadow_array", depth_desc);

    bool executed = false;
    f.graph.add_pass("shadow_pass")
        .color(color)
        .depth(depth, 2)
        .execute([&](WGPURenderPassEncoder) { executed = true; });

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);

    CHECK(executed);
}

TEST_CASE("FrameGraph - color attachment with layer index executes") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.array_layers = 2;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto tex = f.graph.create("color_array", desc);

    bool executed = false;
    f.graph.add_pass("layer_pass").color(tex, 1).execute([&](WGPURenderPassEncoder) {
        executed = true;
    });

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);

    CHECK(executed);
}

TEST_CASE("FrameGraph - array texture cache reuse across frames") {
    TestFixture f;

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.array_layers = 4;
    desc.format = WGPUTextureFormat_Depth32Float;

    // Frame 1
    f.graph.begin_frame();
    auto h1 = f.graph.create("shadow_array", desc);
    f.graph.add_pass("pass").depth(h1, 0).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view1 = f.graph.get_texture_ref(h1).view();

    // Frame 2 — same desc, should reuse
    f.graph.begin_frame();
    auto h2 = f.graph.create("shadow_array", desc);
    f.graph.add_pass("pass").depth(h2, 0).execute([](WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view2 = f.graph.get_texture_ref(h2).view();

    CHECK(view1 == view2);
}

TEST_CASE("FrameGraph - non-array texture has no layer views") {
    TestFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto h = f.graph.create("color", desc);
    f.graph.add_pass("pass").color(h).execute([](WGPURenderPassEncoder) {});

    f.graph.compile();

    auto ref = f.graph.get_texture_ref(h);
    CHECK(ref.view() != nullptr);
    CHECK(ref.layer_count() == 0);
}

// --- Bind group tests ---

namespace {

struct BindGroupFixture : TestFixture {
    WGPUBindGroupLayout create_buffer_layout() {
        WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entry.binding = 0;
        entry.visibility = WGPUShaderStage_Fragment;
        entry.buffer.type = WGPUBufferBindingType_Uniform;
        entry.buffer.minBindingSize = 0;

        WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bgl_desc.entryCount = 1;
        bgl_desc.entries = &entry;
        auto layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);
        REQUIRE(layout != nullptr);
        return layout;
    }

    WGPUBindGroupLayout create_texture_layout() {
        WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entry.binding = 0;
        entry.visibility = WGPUShaderStage_Fragment;
        entry.texture.sampleType = WGPUTextureSampleType_Float;
        entry.texture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor bgl_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bgl_desc.entryCount = 1;
        bgl_desc.entries = &entry;
        auto layout = wgpuDeviceCreateBindGroupLayout(device.handle(), &bgl_desc);
        REQUIRE(layout != nullptr);
        return layout;
    }
};

}  // namespace

TEST_CASE("FrameGraph - bind group with buffer input") {
    BindGroupFixture f;
    auto layout = f.create_buffer_layout();

    f.graph.begin_frame();

    pts::rendering::BufferDesc buf_desc;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    auto buf_h = f.graph.find_or_create_buffer("ubo", buf_desc);

    pts::rendering::BindGroupEntry entry;
    entry.binding = 0;
    entry.buffer = buf_h;

    pts::rendering::BindGroupDesc bg_desc;
    bg_desc.layout = layout;
    bg_desc.entries = {entry};

    auto bg_h = f.graph.find_or_create_bind_group("my_bg", bg_desc);
    CHECK(bg_h.is_valid());

    f.graph.compile();

    auto ref = f.graph.get_bind_group_ref(bg_h);
    CHECK(static_cast<bool>(ref));
    CHECK(ref.handle() != nullptr);

    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - bind group version invalidation on buffer change") {
    BindGroupFixture f;
    auto layout = f.create_buffer_layout();

    WGPUBufferDescriptor ext_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    ext_desc.size = 256;
    ext_desc.usage = WGPUBufferUsage_Uniform;
    auto ext_buf1 = wgpuDeviceCreateBuffer(f.device.handle(), &ext_desc);
    auto ext_buf2 = wgpuDeviceCreateBuffer(f.device.handle(), &ext_desc);
    REQUIRE(ext_buf1 != nullptr);
    REQUIRE(ext_buf2 != nullptr);

    // Frame 1 — import buf1, create bind group
    f.graph.begin_frame();
    auto buf_h = f.graph.import_buffer("ubo", ext_buf1, 256);

    pts::rendering::BindGroupEntry entry;
    entry.binding = 0;
    entry.buffer = buf_h;

    pts::rendering::BindGroupDesc bg_desc;
    bg_desc.layout = layout;
    bg_desc.entries = {entry};

    auto bg_h = f.graph.find_or_create_bind_group("my_bg", bg_desc);
    f.graph.compile();
    auto ref1 = f.graph.get_bind_group_ref(bg_h);
    CHECK(ref1.handle() != nullptr);

    // Frame 2 — import DIFFERENT buffer pointer → version bump → bind group rebuilds
    f.graph.begin_frame();
    auto buf_h2 = f.graph.import_buffer("ubo", ext_buf2, 256);

    pts::rendering::BindGroupEntry entry2;
    entry2.binding = 0;
    entry2.buffer = buf_h2;

    pts::rendering::BindGroupDesc bg_desc2;
    bg_desc2.layout = layout;
    bg_desc2.entries = {entry2};

    auto bg_h2 = f.graph.find_or_create_bind_group("my_bg", bg_desc2);
    f.graph.compile();
    auto ref2 = f.graph.get_bind_group_ref(bg_h2);
    CHECK(ref2.handle() != nullptr);

    // The bind group was rebuilt (different WGPUBindGroup handle)
    CHECK(ref1.handle() != ref2.handle());

    wgpuBufferDestroy(ext_buf1);
    wgpuBufferRelease(ext_buf1);
    wgpuBufferDestroy(ext_buf2);
    wgpuBufferRelease(ext_buf2);
    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - bind group cache reuse when inputs stable") {
    BindGroupFixture f;
    auto layout = f.create_buffer_layout();

    // Frame 1
    f.graph.begin_frame();

    pts::rendering::BufferDesc buf_desc;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    auto buf_h = f.graph.find_or_create_buffer("ubo", buf_desc);

    pts::rendering::BindGroupEntry entry;
    entry.binding = 0;
    entry.buffer = buf_h;

    pts::rendering::BindGroupDesc bg_desc;
    bg_desc.layout = layout;
    bg_desc.entries = {entry};

    f.graph.find_or_create_bind_group("my_bg", bg_desc);
    f.graph.compile();
    auto ref1 = f.graph.get_bind_group_ref(f.graph.find_bind_group("my_bg").value());
    CHECK(ref1.handle() != nullptr);

    // Frame 2 — same buffer desc, same bind group desc → should reuse
    f.graph.begin_frame();
    auto buf_h2 = f.graph.find_or_create_buffer("ubo", buf_desc);

    pts::rendering::BindGroupEntry entry2;
    entry2.binding = 0;
    entry2.buffer = buf_h2;

    pts::rendering::BindGroupDesc bg_desc2;
    bg_desc2.layout = layout;
    bg_desc2.entries = {entry2};

    f.graph.find_or_create_bind_group("my_bg", bg_desc2);
    f.graph.compile();
    auto ref2 = f.graph.get_bind_group_ref(f.graph.find_bind_group("my_bg").value());

    // Same underlying WGPUBindGroup should be reused
    CHECK(ref1.handle() == ref2.handle());

    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - bind group eviction") {
    BindGroupFixture f;
    auto layout = f.create_buffer_layout();

    pts::rendering::BufferDesc buf_desc;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    // Frame 1 — create two bind groups
    f.graph.begin_frame();
    auto buf_a = f.graph.find_or_create_buffer("ubo_a", buf_desc);
    auto buf_b = f.graph.find_or_create_buffer("ubo_b", buf_desc);

    pts::rendering::BindGroupEntry entry_a;
    entry_a.binding = 0;
    entry_a.buffer = buf_a;
    pts::rendering::BindGroupDesc desc_a;
    desc_a.layout = layout;
    desc_a.entries = {entry_a};
    f.graph.find_or_create_bind_group("bg_a", desc_a);

    pts::rendering::BindGroupEntry entry_b;
    entry_b.binding = 0;
    entry_b.buffer = buf_b;
    pts::rendering::BindGroupDesc desc_b;
    desc_b.layout = layout;
    desc_b.entries = {entry_b};
    f.graph.find_or_create_bind_group("bg_b", desc_b);

    f.graph.compile();
    CHECK(f.graph.cached_bind_group_count() == 2);

    // Frame 2 — only use bg_a, bg_b should be evicted
    f.graph.begin_frame();
    auto buf_a2 = f.graph.find_or_create_buffer("ubo_a", buf_desc);

    pts::rendering::BindGroupEntry entry_a2;
    entry_a2.binding = 0;
    entry_a2.buffer = buf_a2;
    pts::rendering::BindGroupDesc desc_a2;
    desc_a2.layout = layout;
    desc_a2.entries = {entry_a2};
    f.graph.find_or_create_bind_group("bg_a", desc_a2);

    f.graph.compile();
    CHECK(f.graph.cached_bind_group_count() == 1);

    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - bind group with texture input") {
    BindGroupFixture f;
    auto layout = f.create_texture_layout();

    pts::rendering::TextureDesc tex_desc;
    tex_desc.width = 64;
    tex_desc.height = 64;
    tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
    tex_desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment;

    // Frame 1 — create texture and bind group referencing it
    f.graph.begin_frame();
    auto tex_h = f.graph.create("my_tex", tex_desc);
    f.graph.add_pass("writer").color(tex_h).execute([](WGPURenderPassEncoder) {});

    pts::rendering::BindGroupEntry entry;
    entry.binding = 0;
    entry.texture = tex_h;

    pts::rendering::BindGroupDesc bg_desc;
    bg_desc.layout = layout;
    bg_desc.entries = {entry};

    auto bg_h = f.graph.find_or_create_bind_group("tex_bg", bg_desc);
    f.graph.compile();
    auto ref1 = f.graph.get_bind_group_ref(bg_h);
    CHECK(ref1.handle() != nullptr);

    // Frame 2 — same texture desc → bind group reused
    f.graph.begin_frame();
    auto tex_h2 = f.graph.create("my_tex", tex_desc);
    f.graph.add_pass("writer").color(tex_h2).execute([](WGPURenderPassEncoder) {});

    pts::rendering::BindGroupEntry entry2;
    entry2.binding = 0;
    entry2.texture = tex_h2;

    pts::rendering::BindGroupDesc bg_desc2;
    bg_desc2.layout = layout;
    bg_desc2.entries = {entry2};

    f.graph.find_or_create_bind_group("tex_bg", bg_desc2);
    f.graph.compile();
    auto ref2 = f.graph.get_bind_group_ref(f.graph.find_bind_group("tex_bg").value());
    CHECK(ref2.handle() != nullptr);
    CHECK(ref1.handle() == ref2.handle());

    // Frame 3 — resize texture → version bump → bind group rebuilds
    tex_desc.width = 128;
    tex_desc.height = 128;

    f.graph.begin_frame();
    auto tex_h3 = f.graph.create("my_tex", tex_desc);
    f.graph.add_pass("writer").color(tex_h3).execute([](WGPURenderPassEncoder) {});

    pts::rendering::BindGroupEntry entry3;
    entry3.binding = 0;
    entry3.texture = tex_h3;

    pts::rendering::BindGroupDesc bg_desc3;
    bg_desc3.layout = layout;
    bg_desc3.entries = {entry3};

    f.graph.find_or_create_bind_group("tex_bg", bg_desc3);
    f.graph.compile();
    auto ref3 = f.graph.get_bind_group_ref(f.graph.find_bind_group("tex_bg").value());
    CHECK(ref3.handle() != nullptr);
    CHECK(ref1.handle() != ref3.handle());

    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - find_bind_group returns nullopt for missing") {
    TestFixture f;
    f.graph.begin_frame();
    CHECK(!f.graph.find_bind_group("nonexistent").has_value());
}

TEST_CASE("FrameGraph - cached_bind_group_count") {
    BindGroupFixture f;
    auto layout = f.create_buffer_layout();

    f.graph.begin_frame();
    CHECK(f.graph.cached_bind_group_count() == 0);

    pts::rendering::BufferDesc buf_desc;
    buf_desc.size = 64;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    auto buf = f.graph.find_or_create_buffer("buf", buf_desc);

    pts::rendering::BindGroupEntry entry;
    entry.binding = 0;
    entry.buffer = buf;

    pts::rendering::BindGroupDesc bg_desc;
    bg_desc.layout = layout;
    bg_desc.entries = {entry};

    f.graph.find_or_create_bind_group("bg", bg_desc);
    f.graph.compile();
    CHECK(f.graph.cached_bind_group_count() == 1);

    wgpuBindGroupLayoutRelease(layout);
}

// --- IPass*-based auto-naming tests ---

#include <core/rendering/shaderLoader.h>

namespace {

struct TestPass : pts::rendering::IPass {
    std::string m_name;
    explicit TestPass(const char* name, const pts::rendering::ShaderLoader& sl)
        : IPass(sl), m_name(name) {
    }
    auto name() const noexcept -> std::string_view override {
        return m_name;
    }
    auto is_ready() const noexcept -> bool override {
        return true;
    }

   protected:
    void do_setup(const pts::webgpu::Device&) override {
    }
};

struct PassFixture : TestFixture {
    pts::rendering::ShaderLoader sl{logger};
    TestPass pass_a{"alpha", sl};
    TestPass pass_b{"beta", sl};
};

}  // namespace

TEST_CASE("FrameGraph - IPass auto-naming creates namespaced keys") {
    PassFixture f;

    f.graph.begin_frame();

    pts::rendering::BufferDesc desc;
    desc.size = 256;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    // Same label from different passes → different resources
    auto h1 = f.graph.find_or_create_buffer(&f.pass_a, desc, "uniforms");
    auto h2 = f.graph.find_or_create_buffer(&f.pass_b, desc, "uniforms");

    CHECK(h1.is_valid());
    CHECK(h2.is_valid());
    CHECK(h1.index != h2.index);

    // Verify they resolve to different cache entries
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 2);
}

TEST_CASE("FrameGraph - IPass auto-naming same pass returns same handle") {
    PassFixture f;

    f.graph.begin_frame();

    pts::rendering::BufferDesc desc;
    desc.size = 128;
    desc.usage = WGPUBufferUsage_Uniform;

    auto h1 = f.graph.find_or_create_buffer(&f.pass_a, desc, "uniforms");
    auto h2 = f.graph.find_or_create_buffer(&f.pass_a, desc, "uniforms");

    CHECK(h1.index == h2.index);
}

TEST_CASE("FrameGraph - IPass auto-naming counter generates unique keys") {
    PassFixture f;

    f.graph.begin_frame();

    pts::rendering::BufferDesc desc;
    desc.size = 64;
    desc.usage = WGPUBufferUsage_Storage;

    // No label → auto-generated keys: alpha/buffer_0, alpha/buffer_1
    auto h1 = f.graph.find_or_create_buffer(&f.pass_a, desc);
    auto h2 = f.graph.find_or_create_buffer(&f.pass_a, desc);

    CHECK(h1.is_valid());
    CHECK(h2.is_valid());
    CHECK(h1.index != h2.index);

    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 2);
}

TEST_CASE("FrameGraph - IPass auto-naming counters reset each frame") {
    PassFixture f;

    pts::rendering::BufferDesc desc;
    desc.size = 64;
    desc.usage = WGPUBufferUsage_Storage;

    // Frame 1
    f.graph.begin_frame();
    f.graph.find_or_create_buffer(&f.pass_a, desc);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 1);

    // Frame 2 — counter resets, same key generated → cache reuse
    f.graph.begin_frame();
    f.graph.find_or_create_buffer(&f.pass_a, desc);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 1);
}

TEST_CASE("FrameGraph - IPass find_or_create texture") {
    PassFixture f;

    f.graph.begin_frame();

    pts::rendering::TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto h1 = f.graph.find_or_create(&f.pass_a, desc, "color");
    auto h2 = f.graph.find_or_create(&f.pass_b, desc, "color");

    CHECK(h1.is_valid());
    CHECK(h2.is_valid());
    CHECK(h1.index != h2.index);
}

TEST_CASE("FrameGraph - IPass import_buffer namespaced") {
    PassFixture f;

    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform;
    auto ext_buf = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    REQUIRE(ext_buf != nullptr);

    f.graph.begin_frame();
    auto h = f.graph.import_buffer(&f.pass_a, ext_buf, 256, "external");
    CHECK(h.is_valid());

    f.graph.compile();
    auto ref = f.graph.get_buffer_ref(h);
    CHECK(ref.handle() == ext_buf);

    wgpuBufferDestroy(ext_buf);
    wgpuBufferRelease(ext_buf);
}

TEST_CASE("FrameGraph - IPass find_or_create_bind_group namespaced") {
    BindGroupFixture f;
    pts::rendering::ShaderLoader sl{f.logger};
    TestPass pass{"test_pass", sl};
    auto layout = f.create_buffer_layout();

    f.graph.begin_frame();

    pts::rendering::BufferDesc buf_desc;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    auto buf_h = f.graph.find_or_create_buffer(&pass, buf_desc, "ubo");

    pts::rendering::BindGroupEntry entry;
    entry.binding = 0;
    entry.buffer = buf_h;

    pts::rendering::BindGroupDesc bg_desc;
    bg_desc.layout = layout;
    bg_desc.entries = {entry};

    auto bg_h = f.graph.find_or_create_bind_group(&pass, std::move(bg_desc), "bg0");
    CHECK(bg_h.is_valid());

    f.graph.compile();
    auto ref = f.graph.get_bind_group_ref(bg_h);
    CHECK(ref.handle() != nullptr);

    wgpuBindGroupLayoutRelease(layout);
}

PTS_TEST_MAIN()
