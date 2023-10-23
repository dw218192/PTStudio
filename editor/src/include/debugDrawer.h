#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>
#include <functional>
#include <list>

#include "utils.h"

struct BoundingBox;
struct Camera;
struct Ray;

struct DebugDrawer {
	static constexpr float k_default_lifetime = 2;
	static constexpr float k_default_thickness = 2;

	void begin_relative(glm::vec2 offset) noexcept;
	void end_relative() noexcept;

	void loop(float dt) noexcept;
	void draw_rect(glm::vec2 min, glm::vec2 max, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_line(glm::vec2 from, glm::vec2 to, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;

	void draw_rect_3d(View<Camera> cam, glm::vec3 center, glm::vec3 extent, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_box(View<Camera> cam, View<BoundingBox> box, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_line_3d(View<Camera> cam, glm::vec3 from, glm::vec3 to, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;
	void draw_ray_3d(View<Camera> cam, View<Ray> ray, glm::vec3 color, float thickness = k_default_thickness, float time = k_default_lifetime) noexcept;

private:
	struct DrawCallInfo {
		float life;
		std::function<void()> draw_call;
	};
	glm::vec2 m_offset = glm::vec2{ 0,0 };
	std::list<DrawCallInfo> m_draw_calls;
};