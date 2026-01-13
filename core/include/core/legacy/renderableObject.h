#pragma once

#include <optional>
#include <string>
#include <tcb/span.hpp>
#include <tl/expected.hpp>

#include "boundingBox.h"
#include "editFlags.h"
#include "light.h"
#include "material.h"
#include "reflection.h"
#include "sceneObject.h"
#include "transform.h"
#include "vertex.h"

namespace PTS {
/**
 * @brief If the object is a primitive, this enum will be used to determine which primitive it is
 */
enum class PrimitiveType { None, Triangle, Quad, Cube, Sphere };

struct Scene;

struct RenderableObject : SceneObject {
    RenderableObject(ObjectConstructorUsage usage = ObjectConstructorUsage::SERIALIZE) noexcept;
    RenderableObject(Scene& scene, std::string_view name, Transform transform, EditFlags edit_flags,
                     Material mat, tcb::span<Vertex const> vertices,
                     tcb::span<unsigned const> indices);
    RenderableObject(Scene& scene, Transform transform, EditFlags edit_flags, Material mat,
                     tcb::span<Vertex const> vertices, tcb::span<unsigned const> indices);

    NODISCARD static auto from_obj(
        Scene& scene, EditFlags edit_flags, Material mat, std::string_view filename,
        std::string* warning = nullptr) noexcept -> tl::expected<RenderableObject, std::string>;
    NODISCARD static auto make_triangle_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                            Transform trans) noexcept -> RenderableObject;
    NODISCARD static auto make_quad_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                        Transform trans) noexcept -> RenderableObject;
    NODISCARD static auto make_cube_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                        Transform trans) noexcept -> RenderableObject;
    NODISCARD static auto make_sphere_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                          Transform trans) noexcept -> RenderableObject;

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

    auto get_proxy_light() const noexcept -> auto const& {
        return m_proxy_light;
    }

    struct FieldTag {
        static constexpr int LOCAL_BOUND = 0;
        static constexpr int MAT = 1;
        static constexpr int VERTICES = 2;
        static constexpr int INDICES = 3;
        static constexpr int PROXY_LIGHT = 4;
        static constexpr int PRIMITIVE_TYPE = 5;
    };

   private:
    DECL_DEFERRED_STATIC_INIT(static_init);
    BEGIN_REFLECT(RenderableObject, SceneObject);
    FIELD(BoundingBox, m_local_bound, {}, MSerialize{}, MNoInspect{});  // not editable
    FIELD(Material, m_mat, {}, MSerialize{});

    FIELD(std::vector<Vertex>, m_vertices, {}, MSerialize{}, MNoInspect{});    // not editable
    FIELD(std::vector<unsigned>, m_indices, {}, MSerialize{}, MNoInspect{});   // not editable
    FIELD(ObserverPtr<Light>, m_proxy_light, {}, MSerialize{}, MNoInspect{});  // not editable
    FIELD(PrimitiveType, m_primitive_type, PrimitiveType::None, MSerialize{},
          MNoInspect{});  // not editable
    END_REFLECT();

    // enables dynamic retrieval of class info for polymorphic types
    DECL_DYNAMIC_INFO();
};
}  // namespace PTS
