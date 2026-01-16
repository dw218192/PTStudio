#include <core/legacy/camera.h>
#include <core/legacy/renderableObject.h>
#include <core/legacy/scene.h>
#include <tiny_obj_loader.h>

#include <glm/ext/scalar_constants.hpp>

PTS::RenderableObject::RenderableObject(ObjectConstructorUsage usage) noexcept
    : SceneObject{usage} {
}

PTS::RenderableObject::RenderableObject(Scene& scene, std::string_view name, Transform transform,
                                        EditFlags edit_flags, Material mat,
                                        tcb::span<Vertex const> vertices,
                                        tcb::span<unsigned const> indices)
    : SceneObject{scene, name, std::move(transform), edit_flags},
      m_mat{std::move(mat)},
      m_vertices{vertices.begin(), vertices.end()},
      m_indices{indices.begin(), indices.end()} {
    m_local_bound = BoundingBox::from_vertices(m_vertices);
}

PTS::RenderableObject::RenderableObject(Scene& scene, Transform transform, EditFlags edit_flags,
                                        Material mat, tcb::span<Vertex const> vertices,
                                        tcb::span<unsigned const> indices)
    : SceneObject{scene, std::move(transform), edit_flags},
      m_mat{std::move(mat)},
      m_vertices{vertices.begin(), vertices.end()},
      m_indices{indices.begin(), indices.end()} {
    m_local_bound = BoundingBox::from_vertices(m_vertices);
}

auto PTS::RenderableObject::from_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                     std::string_view filename, std::string* warning) noexcept
    -> tl::expected<RenderableObject, std::string> {
    std::vector<Vertex> vertices;
    std::vector<unsigned> indices;

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "./";

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename.data(), config)) {
        if (!reader.Error().empty()) {
            return TL_ERROR("Tiny Obj Error:\n" + reader.Error());
        } else {
            return TL_ERROR("Unknown Tiny Obj Error");
        }
    }

    if (!reader.Warning().empty() && warning) {
        *warning = reader.Warning();
    }

    auto const& attrib = reader.GetAttrib();
    // fill vertex positions first, normals and uvs will be filled later
    for (size_t i = 2; i < attrib.vertices.size(); i += 3) {
        Vertex vertex{};
        vertex.position =
            glm::vec3{attrib.vertices[i - 2], attrib.vertices[i - 1], attrib.vertices[i - 0]};
        vertices.emplace_back(vertex);
    }

    for (auto const& s : reader.GetShapes()) {
        for (size_t i = 0; i < s.mesh.material_ids.size(); ++i) {
            auto infer_normals = false;
            for (size_t j = 0; j < 3; ++j) {
                auto vi = s.mesh.indices[3 * i + j].vertex_index;
                auto ni = s.mesh.indices[3 * i + j].normal_index;
                auto uvi = s.mesh.indices[3 * i + j].texcoord_index;

                if (ni != -1) {
                    vertices[vi].normal =
                        glm::vec3{attrib.normals[3 * ni + 0], attrib.normals[3 * ni + 1],
                                  attrib.normals[3 * ni + 2]};
                } else {
                    infer_normals = true;
                }
                if (uvi != -1) {
                    vertices[vi].uv =
                        glm::vec2{attrib.texcoords[2 * uvi + 0], attrib.texcoords[2 * uvi + 1]};
                }
                indices.emplace_back(static_cast<unsigned>(vi));
            }
            if (infer_normals) {
                auto const& v0 = vertices[indices[indices.size() - 3]];
                auto const& v1 = vertices[indices[indices.size() - 2]];
                auto const& v2 = vertices[indices[indices.size() - 1]];
                auto normal = glm::normalize(
                    glm::cross(v1.position - v0.position, v2.position - v0.position));
                vertices[indices[indices.size() - 3]].normal = normal;
                vertices[indices[indices.size() - 2]].normal = normal;
                vertices[indices[indices.size() - 1]].normal = normal;
            }
        }
    }
    return RenderableObject{scene,          Transform{},         edit_flags,
                            std::move(mat), std::move(vertices), std::move(indices)};
}

auto PTS::RenderableObject::make_triangle_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                              Transform trans) noexcept -> RenderableObject {
    std::vector<Vertex> vertices = {
        Vertex{glm::vec3{-0.5, -0.5, 0}, glm::vec3{0, 0, 1}, glm::vec2{0, 0}},
        Vertex{glm::vec3{0.5, -0.5, 0}, glm::vec3{0, 0, 1}, glm::vec2{1, 0}},
        Vertex{glm::vec3{0, 0.5, 0}, glm::vec3{0, 0, 1}, glm::vec2{0.5, 1}}};
    std::vector<unsigned> indices = {0, 1, 2};

    auto ret = RenderableObject{scene,          std::move(trans),    edit_flags,
                                std::move(mat), std::move(vertices), std::move(indices)};
    ret.m_primitive_type = PrimitiveType::Triangle;
    return ret;
}

auto PTS::RenderableObject::make_quad_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                          Transform trans) noexcept -> RenderableObject {
    std::vector<Vertex> vertices = {
        Vertex{glm::vec3{-0.5, -0.5, 0}, glm::vec3{0, 0, 1}, glm::vec2{0, 0}},
        Vertex{glm::vec3{0.5, -0.5, 0}, glm::vec3{0, 0, 1}, glm::vec2{1, 0}},
        Vertex{glm::vec3{-0.5, 0.5, 0}, glm::vec3{0, 0, 1}, glm::vec2{0, 1}},
        Vertex{glm::vec3{0.5, 0.5, 0}, glm::vec3{0, 0, 1}, glm::vec2{1, 1}}};
    std::vector<unsigned> indices = {0, 1, 2, 1, 3, 2};

    auto ret = RenderableObject{scene,          std::move(trans),    edit_flags,
                                std::move(mat), std::move(vertices), std::move(indices)};
    ret.m_primitive_type = PrimitiveType::Quad;
    return ret;
}

auto PTS::RenderableObject::make_cube_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                          Transform trans) noexcept -> RenderableObject {
    std::vector<Vertex> vertices = {
        // front
        Vertex{glm::vec3{-0.5, -0.5, 0.5}, glm::vec3{0, 0, 1}, glm::vec2{0, 0}},
        Vertex{glm::vec3{0.5, -0.5, 0.5}, glm::vec3{0, 0, 1}, glm::vec2{1, 0}},
        Vertex{glm::vec3{-0.5, 0.5, 0.5}, glm::vec3{0, 0, 1}, glm::vec2{0, 1}},
        Vertex{glm::vec3{0.5, 0.5, 0.5}, glm::vec3{0, 0, 1}, glm::vec2{1, 1}},
        // back
        Vertex{glm::vec3{-0.5, -0.5, -0.5}, glm::vec3{0, 0, -1}, glm::vec2{0, 0}},
        Vertex{glm::vec3{0.5, -0.5, -0.5}, glm::vec3{0, 0, -1}, glm::vec2{1, 0}},
        Vertex{glm::vec3{-0.5, 0.5, -0.5}, glm::vec3{0, 0, -1}, glm::vec2{0, 1}},
        Vertex{glm::vec3{0.5, 0.5, -0.5}, glm::vec3{0, 0, -1}, glm::vec2{1, 1}},
        // left
        Vertex{glm::vec3{-0.5, -0.5, -0.5}, glm::vec3{-1, 0, 0}, glm::vec2{0, 0}},
        Vertex{glm::vec3{-0.5, -0.5, 0.5}, glm::vec3{-1, 0, 0}, glm::vec2{1, 0}},
        Vertex{glm::vec3{-0.5, 0.5, -0.5}, glm::vec3{-1, 0, 0}, glm::vec2{0, 1}},
        Vertex{glm::vec3{-0.5, 0.5, 0.5}, glm::vec3{-1, 0, 0}, glm::vec2{1, 1}},
        // right
        Vertex{glm::vec3{0.5, -0.5, -0.5}, glm::vec3{1, 0, 0}, glm::vec2{0, 0}},
        Vertex{glm::vec3{0.5, -0.5, 0.5}, glm::vec3{1, 0, 0}, glm::vec2{1, 0}},
        Vertex{glm::vec3{0.5, 0.5, -0.5}, glm::vec3{1, 0, 0}, glm::vec2{0, 1}},
        Vertex{glm::vec3{0.5, 0.5, 0.5}, glm::vec3{1, 0, 0}, glm::vec2{1, 1}},
        // top
        Vertex{glm::vec3{-0.5, 0.5, -0.5}, glm::vec3{0, 1, 0}, glm::vec2{0, 0}},
        Vertex{glm::vec3{0.5, 0.5, -0.5}, glm::vec3{0, 1, 0}, glm::vec2{1, 0}},
        Vertex{glm::vec3{-0.5, 0.5, 0.5}, glm::vec3{0, 1, 0}, glm::vec2{0, 1}},
        Vertex{glm::vec3{0.5, 0.5, 0.5}, glm::vec3{0, 1, 0}, glm::vec2{1, 1}},
        // bottom
        Vertex{glm::vec3{-0.5, -0.5, -0.5}, glm::vec3{0, -1, 0}, glm::vec2{0, 0}},
        Vertex{glm::vec3{0.5, -0.5, -0.5}, glm::vec3{0, -1, 0}, glm::vec2{1, 0}},
        Vertex{glm::vec3{-0.5, -0.5, 0.5}, glm::vec3{0, -1, 0}, glm::vec2{0, 1}},
        Vertex{glm::vec3{0.5, -0.5, 0.5}, glm::vec3{0, -1, 0}, glm::vec2{1, 1}}};
    std::vector<unsigned> indices = {// front
                                     0, 1, 2, 1, 3, 2,
                                     // back
                                     4, 5, 6, 5, 7, 6,
                                     // left
                                     8, 9, 10, 9, 11, 10,
                                     // right
                                     12, 13, 14, 13, 15, 14,
                                     // top
                                     16, 17, 18, 17, 19, 18,
                                     // bottom
                                     20, 21, 22, 21, 23, 22};

    auto ret = RenderableObject{scene,          std::move(trans),    edit_flags,
                                std::move(mat), std::move(vertices), std::move(indices)};
    ret.m_primitive_type = PrimitiveType::Cube;
    return ret;
}

auto PTS::RenderableObject::make_sphere_obj(Scene& scene, EditFlags edit_flags, Material mat,
                                            Transform trans) noexcept -> RenderableObject {
    // see http://www.songho.ca/opengl/gl_sphere.html
    std::vector<Vertex> vertices;

    // will make these customizable later as a component
    auto radius = 1.0f;
    auto length_inv = 1.0f / radius;
    auto sector_count = 36;
    auto stack_count = 18;
    auto sector_step = 2 * glm::pi<float>() / sector_count;
    auto stack_step = glm::pi<float>() / stack_count;
    for (auto i = 0; i <= stack_count; ++i) {
        auto stack_angle = glm::pi<float>() / 2 - i * stack_step;
        auto xy = radius * glm::cos(stack_angle);
        auto z = radius * glm::sin(stack_angle);

        for (auto j = 0; j <= sector_count; ++j) {
            auto sectorAngle = j * sector_step;
            auto x = xy * cosf(sectorAngle);
            auto y = xy * sinf(sectorAngle);
            auto nx = x * length_inv;
            auto ny = y * length_inv;
            auto nz = z * length_inv;
            auto s = static_cast<float>(j) / sector_count;
            auto t = static_cast<float>(i) / sector_count;
            vertices.emplace_back(
                Vertex{glm::vec3{x, y, z}, glm::vec3{nx, ny, nz}, glm::vec2{s, t}});
        }
    }

    std::vector<unsigned> indices;
    for (auto i = 0; i < stack_count; ++i) {
        auto k1 = i * (sector_count + 1);
        auto k2 = k1 + sector_count + 1;
        for (auto j = 0; j < sector_count; ++j, ++k1, ++k2) {
            if (i != 0) {
                indices.emplace_back(k1);
                indices.emplace_back(k2);
                indices.emplace_back(k1 + 1);
            }
            if (i != (stack_count - 1)) {
                indices.emplace_back(k1 + 1);
                indices.emplace_back(k2);
                indices.emplace_back(k2 + 1);
            }
        }
    }

    auto ret = RenderableObject{scene,          std::move(trans),    edit_flags,
                                std::move(mat), std::move(vertices), std::move(indices)};
    ret.m_primitive_type = PrimitiveType::Sphere;
    return ret;
}

auto PTS::RenderableObject::static_init() -> void {
    get_field_info<1>().get_on_change_callback_list().connect([](auto data) {
        auto& self = data.obj;
        if (data.new_val.is_emissive()) {
            if (!self.m_proxy_light) {
                self.m_proxy_light = self.get_scene()->template emplace_object<Light>(
                    *self.get_scene(), Transform{}, EditFlags::_NoEdit, LightType::Mesh,
                    data.new_val.emission, data.new_val.emission_intensity);
                self.add_child(*self.m_proxy_light);
                self.m_proxy_light->set_transform(Transform{}, TransformSpace::LOCAL);
            } else {
                self.m_proxy_light->set_color(data.new_val.emission);
                self.m_proxy_light->set_intensity(data.new_val.emission_intensity);
            }
        } else {
            if (self.m_proxy_light) {
                self.get_scene()->remove_object(*self.m_proxy_light);
                self.m_proxy_light = nullptr;
            }
        }
    });
}
