#pragma once

#include "transform.h"
#include "boundingBox.h"
#include "material.h"
#include "vertex.h"
#include "utils.h"

#include <tl/expected.hpp>
#include <tcb/span.hpp>
#include <string>

struct Scene;

struct Object {
    Object(Scene const& scene, Transform transform, Material mat, tcb::span<Vertex const> vertices, std::string_view name);
    Object(Scene const& scene, Transform transform, Material mat, tcb::span<Vertex const> vertices);

    NODISCARD static auto from_obj(Scene const& scene, Material mat, std::string_view filename, std::string* warning = nullptr) noexcept -> tl::expected<Object, std::string>;
    NODISCARD static auto make_triangle_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object;
    NODISCARD static auto make_quad_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object;
    NODISCARD static auto make_cube_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object;

    void set_transform(Transform transform) noexcept;
    NODISCARD auto get_transform() const noexcept -> Transform const&;
    void set_name(std::string_view name) noexcept;
    NODISCARD auto get_name() const noexcept -> std::string_view;
    NODISCARD auto get_bound() const noexcept -> BoundingBox const&;
    NODISCARD auto get_vertices() const noexcept -> tcb::span<Vertex const>;
    NODISCARD auto get_material() const noexcept -> Material const&;

private:
    BoundingBox m_local_bound;
    Transform m_transform;
    Material m_mat;
    std::vector<Vertex> m_vertices;
    std::string m_name;
};
