#include "nullImgui.h"

#include <imgui_impl_null.h>

#include <memory>

#include "../imguiBackend.h"

namespace pts::rendering {
namespace {
class NullImguiWindowing final : public IImguiWindowing {
   public:
    explicit NullImguiWindowing(pts::LoggingManager&) {
        ImGui_ImplNull_Init();
    }

    ~NullImguiWindowing() override {
        ImGui_ImplNull_Shutdown();
    }

    void new_frame() override {
        ImGui_ImplNull_NewFrame();
    }
};

class NullImguiRendering final : public IImguiRendering {
   public:
    explicit NullImguiRendering(pts::LoggingManager&) {
    }

    void new_frame() override {
    }
    void render(bool) override {
    }
    void resize() override {
    }
    auto set_render_output(IRenderGraph&) -> ImTextureID override {
        return nullptr;
    }
    void clear_render_output() override {
    }
    [[nodiscard]] auto output_id() const noexcept -> ImTextureID override {
        return nullptr;
    }
};
}  // namespace

auto create_imgui_windowing(IViewport&, pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IImguiWindowing> {
    return std::make_unique<NullImguiWindowing>(logging_manager);
}

auto create_null_imgui_rendering(pts::LoggingManager& logging_manager)
    -> std::unique_ptr<IImguiRendering> {
    return std::make_unique<NullImguiRendering>(logging_manager);
}
}  // namespace pts::rendering
