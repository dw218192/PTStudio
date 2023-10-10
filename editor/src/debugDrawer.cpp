#include "include/debugDrawer.h"
#include "include/boundingBox.h"
#include "include/application.h"

void DebugDrawer::draw_rect(glm::vec2 center, glm::vec2 extent, glm::vec3 color) {
	ImGui::GetForegroundDrawList()->AddRect(
		ImVec2(center.x, center.y - 1), 
		ImVec2(center.x + extent.x, center.y + extent.y),
		ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1)), 0, 0);
}

void DebugDrawer::draw_rect_3d(glm::vec3 center, glm::vec3 extent, glm::vec3 color) {
	glm::vec2 points[8];
	int i = 0;
	for (float x : { -extent.x, extent.x }) {
		for (float y : { -extent.y, extent.y }) {
			for (float z : { -extent.z, extent.z }) {
				points[i++] = glm::vec2{
					Application::get_cam().world_to_viewport(glm::vec3{ x,y,z } + center)
				};
			}
		}
	}

	int faces[6][4] = { {0,2,3,1}, {4,6,7,5}, {0,4,5,1}, {2,6,7,3}, {0,4,6,2}, {1,5,7,3} };
	auto col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 4; ++j) {
			int lst = j == 0 ? 3 : j - 1;
			glm::vec2 from = points[faces[i][lst]];
			glm::vec2 to = points[faces[i][j]];

			ImGui::GetForegroundDrawList()->AddLine(ImVec2(from.x, from.y), ImVec2(to.x, to.y), col);
		}
	}
}
void DebugDrawer::draw_box(BoundingBox const& box, glm::vec3 color) {
	draw_rect_3d(box.get_center(), box.get_extent(), color);
}

void DebugDrawer::draw_line(glm::vec2 from, glm::vec2 to, glm::vec3 color, float thickness) {
	auto col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	ImGui::GetForegroundDrawList()->AddLine(
		ImVec2(from.x, from.y),
		ImVec2(to.x, to.y),
		col,
		thickness
	);
}

void DebugDrawer::draw_line_3d(glm::vec3 from, glm::vec3 to, glm::vec3 color, float thickness) {
	auto frm_2d = Application::get_cam().world_to_viewport(from);
	auto to_2d = Application::get_cam().world_to_viewport(to);
	draw_line(frm_2d, to_2d, color, thickness);
}
