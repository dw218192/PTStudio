#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/shaderCompiler.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/webgpu/device.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <thread>

#include "testApplication.h"

namespace {

using pts::rendering::BufferDeclHandle;
using pts::rendering::BufferDesc;
using pts::rendering::DescriptorDeclHandle;
using pts::rendering::ExecuteContext;
using pts::rendering::FrameGraph;
using pts::rendering::Lifetime;
using pts::rendering::TextureDeclHandle;
using pts::rendering::TextureDesc;

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
    FrameGraph graph{device, logger};

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

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    auto color = f.graph.texture("color", desc);
    f.graph.add_pass("clear_pass").color(color).execute([](ExecuteContext&, WGPURenderPassEncoder) {
    });

    f.graph.compile();

    CHECK(f.graph.compiled_texture(color) != nullptr);
    CHECK(f.graph.compiled_texture(color)->view != nullptr);

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - two-pass Clear then Load") {
    TestFixture f;
    f.graph.begin_frame();

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;

    auto color = f.graph.texture("color", desc);
    f.graph.add_pass("first_pass").color(color).execute([](ExecuteContext&, WGPURenderPassEncoder) {
    });
    f.graph.add_pass("second_pass")
        .color(color)
        .execute([](ExecuteContext&, WGPURenderPassEncoder) {});

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - depth read-only") {
    TestFixture f;
    f.graph.begin_frame();

    TextureDesc color_desc;
    color_desc.width = 64;
    color_desc.height = 64;
    color_desc.format = WGPUTextureFormat_BGRA8Unorm;

    TextureDesc depth_desc;
    depth_desc.width = 64;
    depth_desc.height = 64;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto color0 = f.graph.texture("color0", color_desc);
    auto depth = f.graph.texture("depth", depth_desc);
    auto color1 = f.graph.texture("color1", color_desc);

    f.graph.add_pass("depth_write_pass")
        .color(color0)
        .depth(depth)
        .execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.add_pass("depth_read_pass")
        .color(color1)
        .depth_readonly(depth)
        .execute([](ExecuteContext&, WGPURenderPassEncoder) {});

    f.graph.compile();

    auto encoder = f.create_encoder();
    f.graph.execute(encoder);
    f.submit(encoder);
}

TEST_CASE("FrameGraph - backward dependency throws") {
    TestFixture f;
    f.graph.begin_frame();

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;
    desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;

    TextureDesc depth_desc;
    depth_desc.width = 64;
    depth_desc.height = 64;
    depth_desc.format = WGPUTextureFormat_Depth32Float;

    auto depth_res = f.graph.texture("depth", depth_desc);

    f.graph.add_pass("reader")
        .color(f.graph.texture("color0", desc))
        .depth_readonly(depth_res)
        .execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.add_pass("writer")
        .color(f.graph.texture("color1", desc))
        .depth(depth_res)
        .execute([](ExecuteContext&, WGPURenderPassEncoder) {});

    CHECK_THROWS_AS(f.graph.compile(), std::runtime_error);
}

TEST_CASE("FrameGraph - cache reuse on same desc") {
    TestFixture f;

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    // Frame 1
    f.graph.begin_frame();
    auto d1 = f.graph.texture("color", desc);
    f.graph.add_pass("pass").color(d1).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view1 = f.graph.compiled_texture(d1)->view;

    // Frame 2 - same desc, should reuse
    f.graph.begin_frame();
    auto d2 = f.graph.texture("color", desc);
    f.graph.add_pass("pass").color(d2).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.compile();
    auto view2 = f.graph.compiled_texture(d2)->view;

    CHECK(view1 == view2);
    CHECK(d1 == d2);  // stable handle
}

TEST_CASE("FrameGraph - cache invalidation on resize") {
    TestFixture f;

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    f.graph.begin_frame();
    auto d1 = f.graph.texture("color", desc);
    f.graph.add_pass("pass").color(d1).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.compile();
    CHECK(f.graph.compiled_texture(d1)->view != nullptr);

    // Frame 2 - different size -> decl gets new desc -> compiled recreated.
    // (Normal user pattern would be eviction first; here we force recreation by
    // re-declaring with same name but different width.)
    desc.width = 128;
    desc.height = 128;
    // New name to avoid the width/height mismatch PRECONDITION.
    f.graph.begin_frame();
    auto d2 = f.graph.texture("color_big", desc);
    f.graph.add_pass("pass").color(d2).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.compile();
    CHECK(f.graph.compiled_texture(d2)->view != nullptr);
}

TEST_CASE("FrameGraph - Frame decl eviction when not used next frame") {
    TestFixture f;

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    f.graph.begin_frame();
    f.graph.texture("color_a", desc);
    f.graph.texture("color_b", desc);
    f.graph.add_pass("pass_a")
        .color(f.graph.find_texture("color_a"))
        .execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.add_pass("pass_b")
        .color(f.graph.find_texture("color_b"))
        .execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.compile();
    CHECK(f.graph.cached_texture_count() == 2);

    // Frame 2 - only use color_a, color_b should be evicted
    f.graph.begin_frame();
    auto ha = f.graph.texture("color_a", desc);
    f.graph.add_pass("pass_a").color(ha).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    f.graph.compile();

    CHECK(f.graph.cached_texture_count() == 1);
    CHECK(!f.graph.find_texture("color_b"));
}

TEST_CASE("FrameGraph - Persistent decl survives eviction") {
    TestFixture f;

    // Single 1x1 upload for a persistent texture
    uint8_t pixels[4] = {255, 0, 0, 255};
    WGPUTextureDescriptor tex_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    tex_desc.size = {1, 1, 1};
    tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
    tex_desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst);
    tex_desc.mipLevelCount = 1;
    tex_desc.sampleCount = 1;
    tex_desc.dimension = WGPUTextureDimension_2D;

    f.graph.begin_frame();
    auto persistent = f.graph.texture("persistent", tex_desc, pixels, sizeof(pixels), 4);
    f.graph.compile();
    auto* compiled1 = f.graph.compiled_texture(persistent);
    CHECK(compiled1 != nullptr);

    // Frame 2 - don't reference it. Persistent decls are not evicted.
    f.graph.begin_frame();
    f.graph.compile();

    // Decl still exists
    auto p2 = f.graph.find_texture("persistent");
    CHECK(p2 == persistent);

    // But compiled pointer is only valid during a frame it's been declared
    // (find_texture bumped last_active_frame). After compile, persistent's
    // compiled should be re-populated.
    CHECK(f.graph.compiled_texture(persistent) != nullptr);
    CHECK(f.graph.compiled_texture(persistent) == compiled1);  // same underlying texture
}

TEST_CASE("Cross-frame staleness - persistent decl survives but must be re-touched") {
    // Persistent decls survive across frames but find_texture re-touches them
    // so that the frame graph considers them active for the current frame.
    TestFixture f;

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.format = WGPUTextureFormat_BGRA8Unorm;

    f.graph.begin_frame();
    auto cached_decl = f.graph.texture("pt", desc, Lifetime::Persistent);
    f.graph.add_pass("pass").color(cached_decl).execute([](ExecuteContext&, WGPURenderPassEncoder) {
    });
    f.graph.compile();
    auto frame1 = f.graph.frame_number();
    CHECK(f.graph.compiled_texture(cached_decl) != nullptr);

    // Frame 2 - do NOT re-declare the cached decl.
    f.graph.begin_frame();
    f.graph.compile();
    auto frame2 = f.graph.frame_number();
    CHECK(frame2 == frame1 + 1);

    // After find_texture re-touches it, the handle is still valid.
    auto found = f.graph.find_texture("pt");
    CHECK(found == cached_decl);
    CHECK(f.graph.compiled_texture(found) != nullptr);
}

// --- Buffer tests ---

TEST_CASE("FrameGraph - create buffer, verify compiled") {
    TestFixture f;
    f.graph.begin_frame();

    BufferDesc desc;
    desc.size = 1024;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    auto d = f.graph.buffer("my_buffer", desc);
    CHECK(bool(d));

    f.graph.compile();

    CHECK(f.graph.compiled_buffer(d) != nullptr);
    CHECK(f.graph.compiled_buffer(d)->buffer != nullptr);
    CHECK(f.graph.compiled_buffer(d)->size == 1024);
}

TEST_CASE("FrameGraph - buffer returns same decl on second call") {
    TestFixture f;
    f.graph.begin_frame();

    BufferDesc desc;
    desc.size = 1024;
    desc.usage = WGPUBufferUsage_Storage;

    auto d1 = f.graph.buffer("buf", desc);
    auto d2 = f.graph.buffer("buf", desc);
    CHECK(d1 == d2);
}

TEST_CASE("FrameGraph - buffer larger size triggers realloc") {
    TestFixture f;

    BufferDesc desc;
    desc.size = 512;
    desc.usage = WGPUBufferUsage_Storage;

    f.graph.begin_frame();
    auto d1 = f.graph.buffer("buf", desc);
    f.graph.compile();
    CHECK(f.graph.compiled_buffer(d1)->size == 512);

    // Frame 2 - bigger
    desc.size = 2048;
    f.graph.begin_frame();
    auto d2 = f.graph.buffer("buf", desc);
    f.graph.compile();
    CHECK(f.graph.compiled_buffer(d2)->size == 2048);
    CHECK(d1 == d2);  // stable decl pointer
}

TEST_CASE("FrameGraph - import_buffer same pointer reuses") {
    TestFixture f;

    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform;
    auto ext_buf = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    REQUIRE(ext_buf != nullptr);

    f.graph.begin_frame();
    auto d1 = f.graph.import_buffer("imported", ext_buf, 256, 1);
    f.graph.compile();
    auto* compiled1 = f.graph.compiled_buffer(d1);
    CHECK(f.graph.cached_buffer_count() == 1);

    f.graph.begin_frame();
    auto d2 = f.graph.import_buffer("imported", ext_buf, 256, 1);
    f.graph.compile();
    CHECK(d1 == d2);
    CHECK(f.graph.compiled_buffer(d2) == compiled1);
    CHECK(f.graph.compiled_buffer(d2)->buffer == ext_buf);

    wgpuBufferDestroy(ext_buf);
    wgpuBufferRelease(ext_buf);
}

TEST_CASE("FrameGraph - import_buffer different pointer recreates") {
    TestFixture f;

    WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform;
    auto ext_buf1 = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    auto ext_buf2 = wgpuDeviceCreateBuffer(f.device.handle(), &buf_desc);
    REQUIRE(ext_buf1 != nullptr);
    REQUIRE(ext_buf2 != nullptr);

    f.graph.begin_frame();
    f.graph.import_buffer("imported", ext_buf1, 256, 1);
    f.graph.compile();

    f.graph.begin_frame();
    auto d2 = f.graph.import_buffer("imported", ext_buf2, 256, 2);
    f.graph.compile();
    CHECK(f.graph.compiled_buffer(d2)->buffer == ext_buf2);

    wgpuBufferDestroy(ext_buf1);
    wgpuBufferRelease(ext_buf1);
    wgpuBufferDestroy(ext_buf2);
    wgpuBufferRelease(ext_buf2);
}

TEST_CASE("FrameGraph - buffer eviction when not used") {
    TestFixture f;

    BufferDesc desc;
    desc.size = 512;
    desc.usage = WGPUBufferUsage_Storage;

    f.graph.begin_frame();
    f.graph.buffer("buf_a", desc);
    f.graph.buffer("buf_b", desc);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 2);

    f.graph.begin_frame();
    f.graph.buffer("buf_a", desc);
    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 1);
}

TEST_CASE("FrameGraph - find_buffer") {
    TestFixture f;
    f.graph.begin_frame();

    CHECK(!bool(f.graph.find_buffer("nonexistent")));

    BufferDesc desc;
    desc.size = 128;
    desc.usage = WGPUBufferUsage_Uniform;

    auto d = f.graph.buffer("my_buf", desc);
    CHECK(f.graph.find_buffer("my_buf") == d);
}

// --- Array texture tests ---

TEST_CASE("FrameGraph - array texture creates per-layer views") {
    TestFixture f;
    f.graph.begin_frame();

    TextureDesc desc;
    desc.width = 64;
    desc.height = 64;
    desc.array_layers = 4;
    desc.format = WGPUTextureFormat_Depth32Float;

    auto d = f.graph.texture("shadow_array", desc);
    f.graph.add_pass("shadow0").depth(d, 0).execute([](ExecuteContext&, WGPURenderPassEncoder) {});

    f.graph.compile();

    CHECK(f.graph.compiled_texture(d) != nullptr);
    CHECK(f.graph.compiled_texture(d)->view != nullptr);
    CHECK(f.graph.compiled_texture(d)->layer_views.size() == 4);
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(f.graph.compiled_texture(d)->layer_views[i] != nullptr);
    }
}

// --- Descriptor tests ---

namespace {

struct DescriptorFixture : TestFixture {
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

TEST_CASE("FrameGraph - descriptor with buffer input") {
    DescriptorFixture f;
    auto layout = f.create_buffer_layout();

    f.graph.begin_frame();

    BufferDesc buf_desc;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    auto buf = f.graph.buffer("ubo", buf_desc);

    auto desc = f.graph.descriptor("my_desc", layout).buffer(0, buf).build();
    CHECK(bool(desc));

    f.graph.compile();

    CHECK(f.graph.compiled_descriptor(desc) != nullptr);
    CHECK(f.graph.compiled_descriptor(desc)->bind_group != nullptr);

    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - descriptor rebuilds on buffer change") {
    DescriptorFixture f;
    auto layout = f.create_buffer_layout();

    WGPUBufferDescriptor ext_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    ext_desc.size = 256;
    ext_desc.usage = WGPUBufferUsage_Uniform;
    auto ext_buf1 = wgpuDeviceCreateBuffer(f.device.handle(), &ext_desc);
    auto ext_buf2 = wgpuDeviceCreateBuffer(f.device.handle(), &ext_desc);

    f.graph.begin_frame();
    auto buf = f.graph.import_buffer("ubo", ext_buf1, 256, 1);
    auto desc = f.graph.descriptor("my_desc", layout).buffer(0, buf).build();
    f.graph.compile();
    auto v1 = f.graph.compiled_descriptor(desc)->version;

    f.graph.begin_frame();
    auto buf2 = f.graph.import_buffer("ubo", ext_buf2, 256, 2);
    auto desc2 = f.graph.descriptor("my_desc", layout).buffer(0, buf2).build();
    f.graph.compile();
    CHECK(f.graph.compiled_descriptor(desc2) != nullptr);
    CHECK(f.graph.compiled_descriptor(desc2)->bind_group != nullptr);
    // Version bumps monotonically on rebuild -- proves we did rebuild.
    CHECK(f.graph.compiled_descriptor(desc2)->version != v1);

    wgpuBufferDestroy(ext_buf1);
    wgpuBufferRelease(ext_buf1);
    wgpuBufferDestroy(ext_buf2);
    wgpuBufferRelease(ext_buf2);
    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - descriptor reuses when inputs stable") {
    DescriptorFixture f;
    auto layout = f.create_buffer_layout();

    BufferDesc buf_desc;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    f.graph.begin_frame();
    auto buf = f.graph.buffer("ubo", buf_desc);
    auto desc = f.graph.descriptor("my_desc", layout).buffer(0, buf).build();
    f.graph.compile();
    auto desc1 = f.graph.compiled_descriptor(desc)->bind_group;

    f.graph.begin_frame();
    auto buf2 = f.graph.buffer("ubo", buf_desc);
    auto desc2 = f.graph.descriptor("my_desc", layout).buffer(0, buf2).build();
    f.graph.compile();
    CHECK(f.graph.compiled_descriptor(desc2)->bind_group == desc1);

    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - descriptor eviction") {
    DescriptorFixture f;
    auto layout = f.create_buffer_layout();

    BufferDesc buf_desc;
    buf_desc.size = 256;
    buf_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    f.graph.begin_frame();
    auto buf_a = f.graph.buffer("ubo_a", buf_desc);
    auto buf_b = f.graph.buffer("ubo_b", buf_desc);
    f.graph.descriptor("desc_a", layout).buffer(0, buf_a).build();
    f.graph.descriptor("desc_b", layout).buffer(0, buf_b).build();
    f.graph.compile();
    CHECK(f.graph.cached_descriptor_count() == 2);

    f.graph.begin_frame();
    auto buf_a2 = f.graph.buffer("ubo_a", buf_desc);
    f.graph.descriptor("desc_a", layout).buffer(0, buf_a2).build();
    f.graph.compile();
    CHECK(f.graph.cached_descriptor_count() == 1);

    wgpuBindGroupLayoutRelease(layout);
}

TEST_CASE("FrameGraph - descriptor rebuilds on texture change") {
    DescriptorFixture f;
    auto layout = f.create_texture_layout();

    TextureDesc tex_desc;
    tex_desc.width = 64;
    tex_desc.height = 64;
    tex_desc.format = WGPUTextureFormat_RGBA8Unorm;
    tex_desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment;

    f.graph.begin_frame();
    auto tex = f.graph.texture("my_tex", tex_desc);
    f.graph.add_pass("writer").color(tex).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    auto desc = f.graph.descriptor("tex_desc", layout).texture(0, tex).build();
    f.graph.compile();
    auto v1 = f.graph.compiled_descriptor(desc)->version;
    auto desc1_ptr = f.graph.compiled_descriptor(desc)->bind_group;

    // Frame 2: same desc -> reuse (pointer stable, version stable)
    f.graph.begin_frame();
    auto tex2 = f.graph.texture("my_tex", tex_desc);
    f.graph.add_pass("writer").color(tex2).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    auto desc2 = f.graph.descriptor("tex_desc", layout).texture(0, tex2).build();
    f.graph.compile();
    CHECK(f.graph.compiled_descriptor(desc2)->version == v1);
    CHECK(f.graph.compiled_descriptor(desc2)->bind_group == desc1_ptr);

    // Frame 3: new texture name -> different decl -> descriptor rebuilds.
    f.graph.begin_frame();
    TextureDesc tex3_desc = tex_desc;
    auto tex3 = f.graph.texture("my_tex_v2", tex3_desc);
    f.graph.add_pass("writer").color(tex3).execute([](ExecuteContext&, WGPURenderPassEncoder) {});
    auto desc3 = f.graph.descriptor("tex_desc", layout).texture(0, tex3).build();
    f.graph.compile();
    CHECK(f.graph.compiled_descriptor(desc3)->version != v1);

    wgpuBindGroupLayoutRelease(layout);
}

// --- IPass*-based auto-naming tests ---

namespace {

struct TestPass : pts::rendering::IPass {
    std::string m_name;
    explicit TestPass(const char* name, const pts::rendering::ShaderLoader& sl)
        : IPass(sl), m_name(name) {
    }
    auto name() const noexcept -> std::string_view override {
        return m_name;
    }
};

struct PassFixture : TestFixture {
    pts::rendering::ShaderLoader sl{logger};
    TestPass pass_a{"alpha", sl};
    TestPass pass_b{"beta", sl};
};

}  // namespace

TEST_CASE("FrameGraph - IPass auto-naming creates namespaced decls") {
    PassFixture f;
    f.graph.begin_frame();

    BufferDesc desc;
    desc.size = 256;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    auto d1 = f.graph.buffer(&f.pass_a, desc, "uniforms");
    auto d2 = f.graph.buffer(&f.pass_b, desc, "uniforms");

    CHECK(bool(d1));
    CHECK(bool(d2));
    CHECK(d1 != d2);

    f.graph.compile();
    CHECK(f.graph.cached_buffer_count() == 2);
}

TEST_CASE("FrameGraph - IPass auto-naming same pass returns same decl") {
    PassFixture f;
    f.graph.begin_frame();

    BufferDesc desc;
    desc.size = 128;
    desc.usage = WGPUBufferUsage_Uniform;

    auto d1 = f.graph.buffer(&f.pass_a, desc, "uniforms");
    auto d2 = f.graph.buffer(&f.pass_a, desc, "uniforms");
    CHECK(d1 == d2);
}

// --- FallbackPool ---

#include <core/rendering/fallbackPool.h>

TEST_CASE("FallbackPool - creates color texture view") {
    TestFixture f;
    pts::rendering::FallbackPool pool(f.device);

    auto view = pool.view(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_2D);
    CHECK(view != nullptr);
    auto view2 = pool.view(WGPUTextureFormat_RGBA8Unorm, WGPUTextureViewDimension_2D);
    CHECK(view == view2);
}

TEST_CASE("FrameGraph - sampler pool dedup") {
    TestFixture f;

    auto s1 = f.graph.sampler(WGPUSamplerBindingType_NonFiltering);
    auto s2 = f.graph.sampler(WGPUSamplerBindingType_NonFiltering);
    CHECK(s1 == s2);

    auto s3 = f.graph.sampler(WGPUSamplerBindingType_Filtering);
    CHECK(s3 != s1);
}

// --- Shader cache tests ---

TEST_CASE("FrameGraph - shader() caches by key") {
    TestFixture f;
    pts::rendering::ShaderLoader sl{f.logger};

    auto getter = [](std::string_view key) -> std::optional<std::string_view> {
        if (key == "test/shader.wgsl")
            return "@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0); }";
        return std::nullopt;
    };
    sl.register_shader("test/shader.wgsl", "test/shader.slang", "test/shader.wgsl", getter);

    pts::rendering::EmbeddedCompiler compiler{sl};
    FrameGraph graph{f.device, f.logger, &compiler};

    auto m1 = graph.shader("test/shader.wgsl");
    auto m2 = graph.shader("test/shader.wgsl");

    CHECK(m1 != nullptr);
    CHECK(m1 == m2);
    CHECK(graph.cached_shader_count() == 1);
}

TEST_CASE("FrameGraph - invalidate_shader forces new module") {
    TestFixture f;
    pts::rendering::ShaderLoader sl{f.logger};

    auto getter = [](std::string_view key) -> std::optional<std::string_view> {
        if (key == "test/shader.wgsl")
            return "@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0); }";
        return std::nullopt;
    };
    sl.register_shader("test/shader.wgsl", "test/shader.slang", "test/shader.wgsl", getter);

    pts::rendering::EmbeddedCompiler compiler{sl};
    FrameGraph graph{f.device, f.logger, &compiler};

    graph.shader("test/shader.wgsl");
    CHECK(graph.cached_shader_count() == 1);

    graph.invalidate_shader("test/shader.wgsl");
    CHECK(graph.cached_shader_count() == 0);

    graph.shader("test/shader.wgsl");
    CHECK(graph.cached_shader_count() == 1);
}

// --- Pipeline cache tests ---

namespace {

auto make_pipeline_test_graph(TestFixture& f, pts::rendering::ShaderLoader& sl,
                              pts::rendering::IShaderCompiler& compiler) -> FrameGraph {
    auto getter = [](std::string_view key) -> std::optional<std::string_view> {
        if (key == "test/shader.wgsl")
            return "@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0); }\n"
                   "@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }";
        return std::nullopt;
    };
    sl.register_shader("test/shader.wgsl", "test/shader.slang", "test/shader.wgsl", getter);
    return FrameGraph{f.device, f.logger, &compiler};
}

}  // namespace

TEST_CASE("FrameGraph - render_pipeline returns non-null") {
    TestFixture f;
    pts::rendering::ShaderLoader sl{f.logger};
    pts::rendering::EmbeddedCompiler compiler{sl};
    auto graph = make_pipeline_test_graph(f, sl, compiler);

    auto p = graph.render_pipeline("test_rp")
                 .shader("test/shader.wgsl")
                 .color_format(WGPUTextureFormat_RGBA8Unorm)
                 .build();

    CHECK(p != nullptr);
    CHECK(graph.cached_pipeline_count() == 1);
}

TEST_CASE("FrameGraph - pipeline invalidation on shader hot-reload") {
    TestFixture f;
    FrameGraph graph{f.device, f.logger};

    std::string wgsl_a =
        "@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0); }\n"
        "@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1,0,0,1); }";
    std::string wgsl_b =
        "@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0); }\n"
        "@fragment fn fs_main() -> @location(0) vec4f { return vec4f(0,1,0,1); }";

    auto mod_a = graph.shader_from_wgsl("test_key", wgsl_a);
    auto p1 = graph.render_pipeline("test_rp")
                  .shader_module(mod_a)
                  .color_format(WGPUTextureFormat_RGBA8Unorm)
                  .build();
    CHECK(p1 != nullptr);
    wgpuRenderPipelineAddRef(p1);

    graph.invalidate_shader("test_key");
    auto mod_b = graph.shader_from_wgsl("test_key", wgsl_b);
    CHECK(mod_a != mod_b);

    auto p2 = graph.render_pipeline("test_rp")
                  .shader_module(mod_b)
                  .color_format(WGPUTextureFormat_RGBA8Unorm)
                  .build();
    CHECK(p2 != nullptr);
    CHECK(p1 != p2);

    wgpuRenderPipelineRelease(p1);
}

// --- Persistent texture/buffer with initial upload ---

TEST_CASE("FrameGraph - persistent texture with data") {
    TestFixture f;
    f.graph.begin_frame();

    uint8_t pixels[4] = {255, 0, 128, 255};
    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.size = {1, 1, 1};
    desc.format = WGPUTextureFormat_RGBA8Unorm;
    desc.usage =
        static_cast<WGPUTextureUsage>(WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst);
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.dimension = WGPUTextureDimension_2D;

    auto d1 = f.graph.texture("persistent_tex", desc, pixels, sizeof(pixels), 4);
    f.graph.compile();
    CHECK(f.graph.compiled_texture(d1) != nullptr);
    CHECK(f.graph.compiled_texture(d1)->view != nullptr);
    auto tex1 = f.graph.compiled_texture(d1)->texture;

    f.graph.begin_frame();
    auto d2 = f.graph.texture("persistent_tex", desc, pixels, sizeof(pixels), 4);
    f.graph.compile();
    CHECK(d1 == d2);
    CHECK(f.graph.compiled_texture(d2)->texture == tex1);  // reused
}

TEST_CASE("FrameGraph - persistent buffer with data") {
    TestFixture f;
    f.graph.begin_frame();

    uint32_t data[] = {1, 2, 3, 4};
    BufferDesc desc;
    desc.size = sizeof(data);
    desc.usage = static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

    auto d1 = f.graph.buffer("persistent_buf", desc, data);
    f.graph.compile();
    CHECK(f.graph.compiled_buffer(d1) != nullptr);
    CHECK(f.graph.compiled_buffer(d1)->buffer != nullptr);
    auto b1 = f.graph.compiled_buffer(d1)->buffer;

    f.graph.begin_frame();
    auto d2 = f.graph.buffer("persistent_buf", desc, data);
    f.graph.compile();
    CHECK(d1 == d2);
    CHECK(f.graph.compiled_buffer(d2)->buffer == b1);
}

PTS_TEST_MAIN()
