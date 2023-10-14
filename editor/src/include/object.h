#pragma once

#include "ext.h"
#include "transform.h"
#include "boundingBox.h"
#include "material.h"
#include "vertex.h"

#include <tl/expected.hpp>
#include <string_view>

struct Scene;

struct Object {
    Object(Scene const& scene, Transform transform, Material mat, std::vector<Vertex> vertices, std::string name);
    Object(Scene const& scene, Transform transform, Material mat, std::vector<Vertex> vertices);

    [[nodiscard]] static auto from_obj(Scene const& scene, Material mat, std::string_view filename) noexcept -> tl::expected<Object, std::string>;
    [[nodiscard]] static auto make_triangle_obj(Scene const& scene, Material mat, Transform const& trans) noexcept -> Object;
    [[nodiscard]] static auto make_quad_obj(Scene const& scene, Material mat, Transform const& trans) noexcept -> Object;

    void set_transform(Transform const& transform) noexcept;
    [[nodiscard]] auto get_transform() const noexcept -> Transform const&;
    void set_name(std::string_view name) noexcept;
    [[nodiscard]] auto get_name() const noexcept -> std::string_view;
    [[nodiscard]] auto get_bound() const noexcept -> BoundingBox const&;
    [[nodiscard]] auto get_vertices() const noexcept -> std::vector<Vertex> const&;
    [[nodiscard]] auto get_material() const noexcept -> Material const&;

private:
    BoundingBox m_bound;
    Transform m_transform;
    Material m_mat;
    std::vector<Vertex> m_vertices;
    std::string m_name;
};
