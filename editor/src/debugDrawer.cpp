#include "include/debugDrawer.h"
#include "include/boundingBox.h"
#include "include/ray.h"
#include "include/application.h"

void DebugDrawer::loop(float dt) {
	for(auto it = m_draw_calls.begin(); it != m_draw_calls.end(); ) {
		auto&& call = *it;
		call.draw_call();
		call.life -= dt;
		
		if (call.life <= 0) {
			it = m_draw_calls.erase(it);
		} else {
			++it;
		}
	}
}

void DebugDrawer::draw_rect(glm::vec2 center, glm::vec2 extent, glm::vec3 color, float thickness, float time) noexcept {
	auto col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	m_draw_calls.emplace_back(
		DrawCallInfo{
			time,
			[=]() {
				ImGui::GetForegroundDrawList()->AddRect(
					ImVec2(center.x, center.y - 1), 
					ImVec2(center.x + extent.x, center.y + extent.y),
					col,
					0,
					0,
					thickness
				);
			}
		}
	);
}

void DebugDrawer::draw_line(glm::vec2 from, glm::vec2 to, glm::vec3 color, float thickness, float time) noexcept {
	auto col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	m_draw_calls.emplace_back(
		DrawCallInfo{
			time,
			[=]() {
				ImGui::GetForegroundDrawList()->AddLine(
					ImVec2(from.x, from.y),
					ImVec2(to.x, to.y),
					col,
					thickness
				);
			}
		}
	);
}
void DebugDrawer::draw_rect_3d(glm::vec3 center, glm::vec3 extent, glm::vec3 color, float thickness, float time) noexcept {
	glm::vec3 points[8];
	int i = 0;
	for (float x : { -extent.x, extent.x }) {
		for (float y : { -extent.y, extent.y }) {
			for (float z : { -extent.z, extent.z }) {
				points[i++] = glm::vec3{ x,y,z } + center;
			}
		}
	}

	int faces[6][4] = { {0,2,3,1}, {4,6,7,5}, {0,4,5,1}, {2,6,7,3}, {0,4,6,2}, {1,5,7,3} };
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 4; ++j) {
			int lst = j == 0 ? 3 : j - 1;
			glm::vec3 from = points[faces[i][lst]];
			glm::vec3 to = points[faces[i][j]];
			draw_line_3d(from, to, color, thickness, time);
		}
	}
}
void DebugDrawer::draw_box(BoundingBox const& box, glm::vec3 color, float thickness, float time) noexcept {
	draw_rect_3d(box.get_center(), box.get_extent(), color, thickness, time);
}

void DebugDrawer::draw_line_3d(glm::vec3 from, glm::vec3 to, glm::vec3 color, float thickness, float time) noexcept {
	auto col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	m_draw_calls.emplace_back(
		DrawCallInfo{
			time,
			[=]() {
				// have to fetch these every frame because the camera might have changed
				auto view_frm = Application::get_cam().world_to_viewport(from);
				auto view_to = Application::get_cam().world_to_viewport(to);

				ImGui::GetForegroundDrawList()->AddLine(
					ImVec2(view_frm.x, view_frm.y),
					ImVec2(view_to.x, view_to.y),
					col,
					thickness
				);
			}
		}
	);
}

void DebugDrawer::draw_ray_3d(Ray const& ray, glm::vec3 color, float thickness, float time) noexcept {
	draw_line_3d(ray.origin, ray.origin + ray.direction * 1000.0f, color, thickness, time);
}