#pragma once

#include <core/rendering/frameGraph.h>
#include <core/rendering/renderPass.h>
#include <core/rendering/webgpu/webgpu.h>

#include <cstdint>
#include <string_view>

namespace pts::editor {

class LobePass final : public rendering::IPass {
   public:
    using IPass::IPass;

    LobePass(const LobePass&) = delete;
    LobePass& operator=(const LobePass&) = delete;
    LobePass(LobePass&&) = delete;
    LobePass& operator=(LobePass&&) = delete;

    [[nodiscard]] auto name() const noexcept -> std::string_view override;
    [[nodiscard]] auto requires_viewport() const noexcept -> bool override {
        return false;
    }

    void render(rendering::FrameGraph& fg, const rendering::PassContext& ctx);
    void draw_imgui() override;
    void update_texture_refs(rendering::FrameGraph& fg) override;

    /// Set material parameters from the selected prim's bound material.
    void set_material(float roughness, float metallic);

    /// Draw the lobe visualization widget inline (no ImGui::Begin/End window).
    /// Returns true if roughness or metallic was changed by the user.
    bool draw_lobe_widget();

    [[nodiscard]] float roughness() const {
        return m_roughness;
    }
    [[nodiscard]] float metallic() const {
        return m_metallic;
    }

    static constexpr uint32_t k_texture_size = 256;
    static constexpr uint32_t k_grid_cols = 128;
    static constexpr uint32_t k_grid_rows = 64;
    static constexpr uint32_t k_uniform_align = 256;

   private:
    // Frame graph decls for self-contained ImGui display (cached ref to
    // compiled view for ImGui::Image across frames).
    rendering::TextureDeclHandle m_lobe_color_decl;
    WGPUTextureView m_lobe_color_view = nullptr;

    // ImGui parameters
    float m_roughness = 0.5f;
    float m_metallic = 0.0f;
    float m_scale = 1.0f;
    float m_light_azimuth_deg = 0.0f;
    float m_light_elevation_deg = 45.0f;
    bool m_show_specular = true;
    bool m_show_diffuse = true;
};

}  // namespace pts::editor
