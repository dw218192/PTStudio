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
    explicit Object(struct Scene const& scene);
    Object(Scene const& scene, BoundingBox bound, Transform transform, Material mat, std::vector<Vertex> vertices, std::string name);
    Object(Scene const& scene, BoundingBox bound, Transform transform, Material mat, std::vector<Vertex> vertices);

    [[nodiscard]] static auto from_obj(Scene const& scene, Material mat, std::string_view filename) noexcept -> tl::expected<Object, std::string>;
    [[nodiscard]] static auto make_triangle_obj(Scene const& scene, Material mat, Transform const& trans) noexcept -> Object;
    [[nodiscard]] static auto make_quad_obj(Scene const& scene, Material mat, Transform const& trans) noexcept -> Object;

    BoundingBox bound;
    Transform transform;
    Material material;
    std::vector<Vertex> vertices;
    std::string name;
};
