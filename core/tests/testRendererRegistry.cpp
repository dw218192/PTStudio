#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <core/rendering/rendererRegistry.h>
#include <core/rendering/scenePass.h>

using namespace pts::rendering;

namespace {

struct FakePass final : IScenePass {
    auto name() const noexcept -> std::string_view override { return "fake"; }
    auto is_ready() const noexcept -> bool override { return true; }
    void setup(const pts::webgpu::Device& /*device*/) override {}
    void add_to_frame_graph(FrameGraph& /*fg*/, const PassContext& /*ctx*/) override {}
};

struct AnotherFakePass final : IScenePass {
    auto name() const noexcept -> std::string_view override { return "another"; }
    auto is_ready() const noexcept -> bool override { return true; }
    void setup(const pts::webgpu::Device& /*device*/) override {}
    void add_to_frame_graph(FrameGraph& /*fg*/, const PassContext& /*ctx*/) override {}
};

// Exercise the REGISTER_RENDERER macro at file scope.
REGISTER_RENDERER("Fake", FakePass);
REGISTER_RENDERER("Another", AnotherFakePass);

}  // namespace

TEST_CASE("RendererRegistry::find returns factory for registered renderer") {
    auto factory = RendererRegistry::find("Fake");
    REQUIRE(factory);
    auto pass = factory();
    REQUIRE(pass);
    CHECK(pass->name() == "fake");
}

TEST_CASE("RendererRegistry::find returns correct factory among multiple entries") {
    auto factory = RendererRegistry::find("Another");
    REQUIRE(factory);
    auto pass = factory();
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
