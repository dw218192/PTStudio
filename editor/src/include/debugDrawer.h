#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <string_view>

struct BoundingBox;

namespace DebugDrawer {
	void draw_rect(glm::vec2 center, glm::vec2 extent, glm::vec3 color);
	void draw_rect_3d(glm::vec3 center, glm::vec3 extent, glm::vec3 color);
	void draw_box(BoundingBox const&, glm::vec3 color);
	void draw_line(glm::vec3 from, glm::vec3 to, glm::vec3 color, float thickness = 2);
	bool draw_img(std::string_view filename, int w, int h);
}