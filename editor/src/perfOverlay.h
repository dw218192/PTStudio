#pragma once

#include <core/rendering/renderWorld.h>
#include <core/rendering/scenePass.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace pts::editor {

struct PerfOverlay {
    static constexpr size_t k_history_size = 128;
    static constexpr float k_ema_alpha = 0.05f;

    void draw(float dt, const rendering::RenderWorld& world, const rendering::FrameGraph& fg,
              const std::vector<rendering::IScenePass*>& passes, std::string_view renderer_name,
              uint32_t viewport_w, uint32_t viewport_h) {
        update_timing(dt);

        ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Performance", nullptr, ImGuiWindowFlags_NoFocusOnAppearing)) {
            ImGui::End();
            return;
        }

        draw_timing_section();
        draw_scene_section(world);
        draw_renderer_section(fg, passes, renderer_name, viewport_w, viewport_h);

        ImGui::End();
    }

   private:
    void update_timing(float dt) {
        m_history[m_history_offset] = dt * 1000.0f;
        m_history_offset = (m_history_offset + 1) % k_history_size;
        if (m_history_count < k_history_size) ++m_history_count;

        if (m_ema_frame_time < 0.0f) {
            m_ema_frame_time = dt;
        } else {
            m_ema_frame_time = k_ema_alpha * dt + (1.0f - k_ema_alpha) * m_ema_frame_time;
        }
    }

    void draw_timing_section() const {
        if (!ImGui::CollapsingHeader("Frame Timing", ImGuiTreeNodeFlags_DefaultOpen)) return;

        float fps = m_ema_frame_time > 0.0f ? 1.0f / m_ema_frame_time : 0.0f;
        float ms = m_ema_frame_time * 1000.0f;

        ImGui::Text("FPS:  %.1f", fps);
        ImGui::Text("Frame: %.2f ms", ms);

        // Build contiguous array for PlotLines (ring buffer unwrap)
        std::array<float, k_history_size> ordered{};
        size_t count = m_history_count;
        for (size_t i = 0; i < count; ++i) {
            size_t src = (m_history_offset + k_history_size - count + i) % k_history_size;
            ordered[i] = m_history[src];
        }

        float max_val = *std::max_element(ordered.begin(), ordered.begin() + count);
        float scale_max = std::max(max_val * 1.2f, 1.0f);

        ImGui::PlotLines("##frame_ms", ordered.data(), static_cast<int>(count), 0, nullptr, 0.0f,
                         scale_max, ImVec2(-1, 40));
    }

    void draw_scene_section(const rendering::RenderWorld& world) const {
        if (!ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) return;

        auto objects = world.get_objects();
        auto meshes = world.get_meshes();
        auto lights = world.get_lights();
        auto materials = world.get_materials();

        uint32_t active_objects = 0;
        uint32_t total_triangles = 0;
        uint32_t active_lights = 0;

        for (auto& obj : objects) {
            if (!obj.active) continue;
            ++active_objects;
            if (obj.mesh_index < meshes.size()) {
                total_triangles += meshes[obj.mesh_index].index_count / 3;
            }
        }
        for (auto& light : lights) {
            if (light.active) ++active_lights;
        }

        ImGui::Text("Objects:   %u / %u", active_objects, static_cast<uint32_t>(objects.size()));
        ImGui::Text("Triangles: %u", total_triangles);
        ImGui::Text("Lights:    %u / %u", active_lights, static_cast<uint32_t>(lights.size()));
        ImGui::Text("Materials: %u", static_cast<uint32_t>(materials.size()));
    }

    void draw_renderer_section(const rendering::FrameGraph& fg,
                               const std::vector<rendering::IScenePass*>& passes,
                               std::string_view renderer_name, uint32_t viewport_w,
                               uint32_t viewport_h) const {
        if (!ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen)) return;

        ImGui::Text("Config: %.*s", static_cast<int>(renderer_name.size()), renderer_name.data());
        ImGui::Text("Viewport: %ux%u", viewport_w, viewport_h);
        ImGui::Text("Cached textures: %u", static_cast<uint32_t>(fg.cached_texture_count()));

        ImGui::Text("Passes:");
        for (auto& pass : passes) {
            ImGui::BulletText("%.*s %s", static_cast<int>(pass->name().size()), pass->name().data(),
                              pass->is_ready() ? "" : "(not ready)");
        }
    }

    std::array<float, k_history_size> m_history{};
    size_t m_history_offset = 0;
    size_t m_history_count = 0;
    float m_ema_frame_time = -1.0f;
};

}  // namespace pts::editor
