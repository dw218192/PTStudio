#pragma once

#include <tl/expected.hpp>
#include <tcb/span.hpp>
#include <string>

#include "transform.h"
#include "material.h"
#include "vertex.h"
#include "utils.h"
#include "reflection.h"
#include "boundingBox.h"

/**
 * \brief If the object is a primitive, this enum will be used to determine which primitive it is
*/
enum class PrimitiveType {
    None,
    Triangle,
    Quad,
    Cube,
    Sphere
};

struct Scene;

struct Object {
    Object() noexcept = default;
    Object(Scene const& scene, Transform transform, Material mat, tcb::span<Vertex const> vertices, std::string_view name);
    Object(Scene const& scene, Transform transform, Material mat, tcb::span<Vertex const> vertices);
    Object(Transform transform, Material mat, tcb::span<Vertex const> vertices, std::string_view name);
    Object(Transform transform, Material mat, tcb::span<Vertex const> vertices);

    NODISCARD static auto from_obj(Scene const& scene, Material mat, std::string_view filename, std::string* warning = nullptr) noexcept -> tl::expected<Object, std::string>;
    NODISCARD static auto make_triangle_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object;
    NODISCARD static auto make_quad_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object;
    NODISCARD static auto make_cube_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object;
    NODISCARD static auto make_sphere_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object;

    NODISCARD auto get_transform() const noexcept -> auto const& {
        return m_transform;
    }
    NODISCARD auto get_name() const noexcept -> std::string_view {
        return m_name;
    }
    NODISCARD auto get_bound() const noexcept -> auto const& {
        return m_local_bound;
    }
    NODISCARD auto get_vertices() const noexcept -> tcb::span<Vertex const> {
        return m_vertices;
    }
    NODISCARD auto get_material() const noexcept -> auto const& {
        return m_mat;
    }
    auto set_transform(Transform transform) noexcept -> void {
        m_transform = std::move(transform);
    }
    auto set_name(std::string_view name) noexcept -> void {
        m_name = name;
    }
    auto set_material(Material mat) noexcept -> void {
        m_mat = std::move(mat);
    }

    auto is_primitive() const noexcept -> bool {
        return m_primitive_type != PrimitiveType::None;
    }
    auto get_primitive_type() const noexcept -> PrimitiveType {
        return m_primitive_type;
    }
private:
    BEGIN_REFLECT(Object);
        FIELD_MOD(BoundingBox, m_local_bound, {},
                MSerialize{});
        FIELD_MOD(Transform, m_transform, {},
                MSerialize{});
        FIELD_MOD(Material, m_mat, {},
                MSerialize{});
        FIELD_MOD(std::vector<Vertex>, m_vertices, {},
                MSerialize{});
        FIELD_MOD(std::string, m_name, "Object",
                MSerialize{});
        FIELD_MOD(PrimitiveType, m_primitive_type, PrimitiveType::None,
                MSerialize{});
    END_REFLECT();
};
