#pragma once

#include <tl/expected.hpp>
#include <tcb/span.hpp>
#include <string>

#include "transform.h"
#include "boundingBox.h"
#include "material.h"
#include "vertex.h"
#include "utils.h"
#include "reflection.h"

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

    NODISCARD auto get_transform() const noexcept -> Transform const&;
    NODISCARD auto get_name() const noexcept -> std::string_view;
    NODISCARD auto get_bound() const noexcept -> BoundingBox const&;
    NODISCARD auto get_vertices() const noexcept -> tcb::span<Vertex const>;
    NODISCARD auto get_material() const noexcept -> Material const&;
    void set_transform(Transform transform) noexcept;
    void set_name(std::string_view name) noexcept;
    void set_material(Material mat) noexcept;
    
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
    END_REFLECT();
};
