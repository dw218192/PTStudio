#pragma once

#include <tl/expected.hpp>
#include <tcb/span.hpp>
#include <string>
#include <optional>

#include "transform.h"
#include "material.h"
#include "vertex.h"
#include "sceneObject.h"
#include "light.h"
#include "reflection.h"
#include "boundingBox.h"
#include "editFlags.h"

namespace PTS {
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

    struct RenderableObject : SceneObject {
        RenderableObject() noexcept = default;
        RenderableObject(Scene const& scene, std::string_view name, Transform transform, Material mat, tcb::span<Vertex const> vertices, tcb::span<unsigned const> indices);
        RenderableObject(Scene const& scene, Transform transform, Material mat, tcb::span<Vertex const> vertices, tcb::span<unsigned const> indices);

        NODISCARD static auto from_obj(Scene const& scene, Material mat, std::string_view filename, std::string* warning = nullptr) noexcept
            -> tl::expected<RenderableObject, std::string>;
        NODISCARD static auto make_triangle_obj(Scene const& scene, Material mat, Transform trans) noexcept -> RenderableObject;
        NODISCARD static auto make_quad_obj(Scene const& scene, Material mat, Transform trans) noexcept -> RenderableObject;
        NODISCARD static auto make_cube_obj(Scene const& scene, Material mat, Transform trans) noexcept -> RenderableObject;
        NODISCARD static auto make_sphere_obj(Scene const& scene, Material mat, Transform trans) noexcept -> RenderableObject;

        NODISCARD auto get_bound() const noexcept -> auto const& {
            return m_local_bound;
        }
        NODISCARD auto get_vertices() const noexcept -> tcb::span<Vertex const> {
            return m_vertices;
        }
        NODISCARD auto get_indices() const noexcept -> tcb::span<unsigned const> {
            return m_indices;
        }
        NODISCARD auto get_material() const noexcept -> auto const& {
            return m_mat;
        }
        auto get_primitive_type() const noexcept -> PrimitiveType {
            return m_primitive_type;
        }
        auto is_primitive() const noexcept -> bool {
            return m_primitive_type != PrimitiveType::None;
        }
        auto set_material(Material mat) noexcept -> void;
    private:
        BEGIN_REFLECT_INHERIT(RenderableObject, SceneObject);
        FIELD_MOD(BoundingBox, m_local_bound, {},
            MSerialize{}, MNoInspect{}); // not editable
        FIELD_MOD(Material, m_mat, {},
            MSerialize{});
        FIELD_MOD(std::vector<Vertex>, m_vertices, {},
            MSerialize{}, MNoInspect{}); // not editable
        FIELD_MOD(std::vector<unsigned>, m_indices, {},
            MSerialize{}, MNoInspect{}); // not editable
        FIELD_MOD(PrimitiveType, m_primitive_type, PrimitiveType::None,
            MSerialize{}, MNoInspect{}); // not editable
        END_REFLECT_INHERIT();
    };
}