#pragma once
#include "ext.h"
#include "result.h"
#include "shader.h"
#include "transform.h"

#include <array>
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
    Object(ShaderProgram const& shader, Transform transform);
    [[nodiscard]] auto from_obj(std::string_view filename) noexcept -> Result<void>;
    [[nodiscard]] auto vertices() const noexcept -> std::vector<Vertex> const& { return m_vertices; }
    [[nodiscard]] static auto make_triangle_obj(ShaderProgram const& shader, Transform const& trans) noexcept -> Object;
    [[nodiscard]] auto bound() const noexcept -> AABB const& { return m_bound; }
    [[nodiscard]] auto begin_draw(struct Camera const& cam) const noexcept -> Result<void>;
    void end_draw() const noexcept;

private:
    AABB m_bound;
    Transform m_transform;
    ShaderProgram const* m_program;
    std::vector<Vertex> m_vertices;
};
