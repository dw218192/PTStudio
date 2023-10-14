#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>
#include <functional>
#include <list>

struct BoundingBox;
struct Ray;

struct DebugDrawer {
	static constexpr float k_default_lifetime = 2;
	static constexpr float k_default_thickness = 2;

	void loop(float dt);
	void draw_rect(glm::vec2 center, glm::vec2 extent, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_line(glm::vec2 from, glm::vec2 to, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;

	void draw_rect_3d(glm::vec3 center, glm::vec3 extent, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_box(BoundingBox const& box, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_line_3d(glm::vec3 from, glm::vec3 to, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_ray_3d(Ray const& ray, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;

private:
	struct DrawCallInfo {
		float life;
		std::function<void()> draw_call;
	};
	std::list<DrawCallInfo> m_draw_calls;
};