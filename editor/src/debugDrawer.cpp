#include "debugDrawer.h"

#include <core/imgui/imhelper.h>
#include <core/legacy/boundingBox.h>
#include <core/legacy/camera.h>
#include <core/legacy/ray.h>

using namespace PTS;

void DebugDrawer::begin_relative(glm::vec2 offset) noexcept {
    m_offset = offset;
}

void DebugDrawer::end_relative() noexcept {
    m_offset = glm::vec2{0, 0};
}

void DebugDrawer::loop(Ref<Application> app, float dt) noexcept {
    for (auto it = m_draw_calls.begin(); it != m_draw_calls.end();) {
        auto&& call = *it;
        call.draw_call(app);
        call.life -= dt;

        if (call.life <= 0) {
            it = m_draw_calls.erase(it);
        } else {
            ++it;
        }
    }
}

void DebugDrawer::draw_rect(glm::vec2 min, glm::vec2 max, glm::vec3 color, float thickness,
                            float time) noexcept {
    min += m_offset;
    max += m_offset;

    auto const col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
    m_draw_calls.emplace_back(DrawCallInfo{time, [=](Ref<Application>) {
                                               ImGui::GetForegroundDrawList()->AddRect(
                                                   ImVec2(min.x, min.y), ImVec2(max.x, max.y), col,
                                                   0, 0, thickness);
                                           }});
}

void DebugDrawer::draw_line(glm::vec2 from, glm::vec2 to, glm::vec3 color, float thickness,
                            float time) noexcept {
    from += m_offset;
    to += m_offset;

    auto const col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
    m_draw_calls.emplace_back(DrawCallInfo{time, [=](Ref<Application>) {
                                               ImGui::GetForegroundDrawList()->AddLine(
                                                   ImVec2(from.x, from.y), ImVec2(to.x, to.y), col,
                                                   thickness);
                                           }});
}

void DebugDrawer::draw_img(glm::vec2 pos, glm::vec2 size, TextureHandle tex, float time) noexcept {
    pos += m_offset;

    m_draw_calls.emplace_back(DrawCallInfo{time, [=](Ref<Application> app) {
                                               if (auto res = tex->bind(); !res) {
                                                   app.get().log(pts::LogLevel::Error, res.error());
                                                   return;
                                               }
                                               ImGui::GetForegroundDrawList()->AddImage(
                                                   tex->get_id(), ImVec2(pos.x, pos.y),
                                                   ImVec2(pos.x + size.x, pos.y + size.y));
                                               tex->unbind();
                                           }});
}

void DebugDrawer::draw_rect_3d(View<Camera> cam, glm::ivec2 vp_size, glm::vec3 center,
                               glm::vec3 extent, glm::vec3 color, float thickness,
                               float time) noexcept {
    glm::vec3 points[8];
    int i = 0;
    for (float x : {-extent.x, extent.x}) {
        for (float y : {-extent.y, extent.y}) {
            for (float z : {-extent.z, extent.z}) {
                points[i++] = glm::vec3{x, y, z} + center;
            }
        }
    }

    int faces[6][4] = {{0, 2, 3, 1}, {4, 6, 7, 5}, {0, 4, 5, 1},
                       {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 5, 7, 3}};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 4; ++j) {
            int lst = j == 0 ? 3 : j - 1;
            glm::vec3 from = points[faces[i][lst]];
            glm::vec3 to = points[faces[i][j]];
            draw_line_3d(cam, vp_size, from, to, color, thickness, time);
        }
    }
}
void DebugDrawer::draw_box(View<Camera> cam, glm::ivec2 vp_size, View<BoundingBox> box,
                           glm::vec3 color, float thickness, float time) noexcept {
    draw_rect_3d(cam, vp_size, box.get().get_center(), box.get().get_extent(), color, thickness,
                 time);
}

void DebugDrawer::draw_line_3d(View<Camera> cam, glm::ivec2 vp_size, glm::vec3 from, glm::vec3 to,
                               glm::vec3 color, float thickness, float time) noexcept {
    auto const col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
    m_draw_calls.emplace_back(DrawCallInfo{
        time, [=, offset = m_offset](Ref<Application>) {
            // have to fetch these every frame because the camera might have changed
            auto view_frm = cam.get().world_to_viewport(from, vp_size);
            auto view_to = cam.get().world_to_viewport(to, vp_size);

            view_frm += offset;
            view_to += offset;

            ImGui::GetForegroundDrawList()->AddLine(ImVec2(view_frm.x, view_frm.y),
                                                    ImVec2(view_to.x, view_to.y), col, thickness);
        }});
}

void DebugDrawer::draw_ray_3d(View<Camera> cam, glm::ivec2 vp_size, View<Ray> ray, glm::vec3 color,
                              float thickness, float time) noexcept {
    draw_line_3d(cam, vp_size, ray.get().origin, ray.get().get_point(100.0f), color, thickness,
                 time);
}

void DebugDrawer::draw_img_3d(View<Camera> cam, glm::ivec2 vp_size, glm::vec3 pos, glm::vec2 size,
                              TextureHandle tex, float time) noexcept {
    m_draw_calls.emplace_back(DrawCallInfo{time, [=, offset = m_offset](Ref<Application>) {
                                               auto view_pos =
                                                   cam.get().world_to_viewport(pos, vp_size);
                                               view_pos += offset;
                                               draw_img(view_pos, size, tex, time);
                                           }});
}
