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

    [[nodiscard]] static auto from_obj(Material mat, std::string_view filename) noexcept -> tl::expected<Object, std::string>;
    [[nodiscard]] static auto make_triangle_obj(Material mat, Transform const& trans) noexcept -> Object;
    [[nodiscard]] static auto make_quad_obj(Material mat, Transform const& trans) noexcept -> Object;

    BoundingBox bound;
    Transform transform;
    Material material;
    std::vector<Vertex> vertices;
    std::string name;
};
