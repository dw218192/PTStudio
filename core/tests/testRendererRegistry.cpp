#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/frameGraph.h>
#include <core/rendering/renderer.h>
#include <core/rendering/rendererRegistry.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <doctest/doctest.h>
#include <spdlog/spdlog.h>

using namespace pts::rendering;

namespace {

static ShaderLoader make_test_shader_loader() {
    return ShaderLoader(spdlog::default_logger());
}
static auto s_test_sl = make_test_shader_loader();

struct FakePass final : IRenderer {
    using IRenderer::IRenderer;
    auto name() const noexcept -> std::string_view override {
        return "fake";
    }
    HdrOutputs do_add_to_frame_graph(FrameGraph& /*fg*/, const PassContext& /*ctx*/) override {
        return {};
    }
};

struct AnotherFakePass final : IRenderer {
    using IRenderer::IRenderer;
    auto name() const noexcept -> std::string_view override {
        return "another";
    }
    HdrOutputs do_add_to_frame_graph(FrameGraph& /*fg*/, const PassContext& /*ctx*/) override {
        return {};
    }
};

/// A minimal IPass child (not a renderer -- has no children of its own).
struct FakeChild final : IPass {
    using IPass::IPass;
    auto name() const noexcept -> std::string_view override {
        return "fake_child";
    }
    void draw_imgui() override {
        ++imgui_count;
    }

    int imgui_count = 0;
};

// Exercise the REGISTER_RENDERER macro at file scope.
REGISTER_RENDERER("Fake", FakePass);
REGISTER_RENDERER("Another", AnotherFakePass);

}  // namespace

TEST_CASE("RendererRegistry::find returns factory for registered renderer") {
    auto factory = RendererRegistry::find("Fake");
    REQUIRE(factory);
    auto pass = factory(s_test_sl);
    REQUIRE(pass);
    CHECK(pass->name() == "fake");
}

TEST_CASE("RendererRegistry::find returns correct factory among multiple entries") {
    auto factory = RendererRegistry::find("Another");
    REQUIRE(factory);
    auto pass = factory(s_test_sl);
    REQUIRE(pass);
    CHECK(pass->name() == "another");
}

TEST_CASE("RendererRegistry::entries contains all registered renderers") {
    auto& entries = RendererRegistry::entries();
    // At least the two we registered above
    CHECK(entries.size() >= 2);

    bool found_fake = false;
    bool found_another = false;
    for (auto& e : entries) {
        if (e.name == "Fake") found_fake = true;
        if (e.name == "Another") found_another = true;
    }
    CHECK(found_fake);
    CHECK(found_another);
}

TEST_CASE("IRenderer::add_pass returns reference and owns child") {
    FakePass renderer{s_test_sl};
    auto& child = renderer.add_pass<FakeChild>(s_test_sl);
    CHECK(child.name() == "fake_child");
}

// draw_imgui forwarding is exercised at runtime -- ImGui widget state
// makes it impractical to unit-test without a full render backend.
