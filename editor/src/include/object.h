#pragma once
#include "ext.h"
#include "shader.h"
#include "transform.h"
#include "boundingBox.h"
#include "material.h"

#include <tl/expected.hpp>
#include <string_view>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Object {
    Object();
    Object(Material mat);

    [[nodiscard]] auto get_transform() const noexcept -> Transform const& { return m_transform; }
    [[nodiscard]] auto get_material() const noexcept -> Material const& { return m_material; }
	[[nodiscard]] auto get_vertices() const noexcept -> std::vector<Vertex> const& { return m_vertices; }
	[[nodiscard]] auto get_bound() const noexcept -> BoundingBox const& { return m_bound; }

    [[nodiscard]] static auto from_obj(Material mat, std::string_view filename) noexcept -> tl::expected<Object, std::string>;
    [[nodiscard]] static auto make_triangle_obj(Material mat, Transform const& trans) noexcept -> Object;
    [[nodiscard]] static auto make_quad_obj(Material mat, Transform const& trans) noexcept -> Object;

private:
    BoundingBox m_bound;
    Transform m_transform;
    Material m_material;
    std::vector<Vertex> m_vertices;
};
