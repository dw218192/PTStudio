#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>

struct BoundingBox;

namespace DebugDrawer {
	void set_draw_list(ImDrawList* draw_list);
	auto get_draw_list() -> ImDrawList*;
	void draw_rect(glm::vec2 center, glm::vec2 extent, glm::vec3 color, float thickness = 2);
	void draw_rect_3d(glm::vec3 center, glm::vec3 extent, glm::vec3 color, float thickness = 2);
	void draw_box(BoundingBox const& box, glm::vec3 color, float thickness = 2);
	void draw_line(glm::vec2 from, glm::vec2 to, glm::vec3 color, float thickness = 2);
	void draw_line_3d(glm::vec3 from, glm::vec3 to, glm::vec3 color, float thickness = 2);
}