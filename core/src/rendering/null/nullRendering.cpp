#include "nullRendering.h"

namespace pts::rendering {
namespace {
class NullRenderGraph final : public IRenderGraph {
   public:
    explicit NullRenderGraph(pts::LoggingManager&) {
    }

    void resize(uint32_t, uint32_t) override {
    }
    void set_current() override {
    }
    void clear_current() override {
    }
    [[nodiscard]] auto output_texture() const noexcept -> PtsTexture override {
        return PtsTexture{};
    }
    [[nodiscard]] auto api() const noexcept -> const PtsRenderGraphApi* override {
        return nullptr;
    }
};
}  // namespace

auto create_null_render_graph(pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IRenderGraph> {
    return std::make_unique<NullRenderGraph>(logging_manager);
}
}  // namespace pts::rendering
