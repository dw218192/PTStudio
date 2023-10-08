#pragma once
#include "ext.h"
#include "shader.h"
#include "transform.h"
#include <tl/expected.hpp>
#include <string_view>

struct AABB {
    glm::vec3 min_pos, max_pos;

    [[nodiscard]] glm::vec3 get_center() const noexcept {
        return (min_pos + max_pos) * 0.5f;
    }
    [[nodiscard]] glm::vec3 get_extent() const noexcept {
        return (max_pos - min_pos) * 0.5f;
    }
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Object {
	[[nodiscard]] auto vertices() const noexcept -> std::vector<Vertex> const& { return m_vertices; }
	[[nodiscard]] auto bound() const noexcept -> AABB const& { return m_bound; }
    [[nodiscard]] auto begin_draw(struct Camera const& cam) const noexcept -> tl::expected<void, std::string>;
    void end_draw() const noexcept;

    [[nodiscard]] static auto from_obj(ShaderProgramRef shader_prog, std::string_view filename) noexcept -> tl::expected<Object, std::string>;
    [[nodiscard]] static auto make_triangle_obj(ShaderProgramRef shader_prog, Transform const& trans) noexcept -> Object;

private:
    Object(ShaderProgramRef shader_prog);
    AABB m_bound;
    Transform m_transform;
    ShaderProgramRef m_shader_prog;
    std::vector<Vertex> m_vertices;
};
