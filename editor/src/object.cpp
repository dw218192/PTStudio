#include "include/object.h"
#include "include/camera.h"

#include <iostream>

#include "include/scene.h"


Object::Object(Scene const& scene, Transform transform, Material mat, std::vector<Vertex> vertices, std::string name) 
    : m_mat { std::move(mat) },  m_vertices { std::move(vertices) }, m_name { std::move(name) } 
{
    m_bound = BoundingBox::from_vertices(m_vertices);
    set_transform(transform);
}

Object::Object(Scene const& scene, Transform transform, Material mat, std::vector<Vertex> vertices)
    : m_mat { std::move(mat) }, m_vertices { std::move(vertices) }, m_name { scene.next_obj_name() } 
{
    m_bound = BoundingBox::from_vertices(m_vertices);
    set_transform(transform);
}

auto Object::from_obj(Scene const& scene, Material mat, std::string_view filename) noexcept -> tl::expected<Object, std::string> {
    std::vector<Vertex> vertices;

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "./";

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename.data(), config)) {
        if (!reader.Error().empty()) {
            return tl::unexpected{ "Tiny Obj Error:\n" + reader.Error() };
        }
        else {
            return tl::unexpected{ "Unknown Tiny Obj Error" };
        }
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader:\n" << reader.Warning();
    }

    auto const& attrib = reader.GetAttrib();
    bool infer_normal = false;
    for (auto const& s : reader.GetShapes()) {
        auto const& indices = s.mesh.indices;
        for (size_t i = 0; i < s.mesh.material_ids.size(); ++i) {
            for (size_t j = 0; j < 3; ++j) {
                int vi = indices[3 * i + j].vertex_index;
                int ni = indices[3 * i + j].normal_index;
                int uvi = indices[3 * i + j].texcoord_index;

                Vertex vertex;
                vertex.position = glm::vec3{ 
                    attrib.vertices[3 * vi + 0], 
                    attrib.vertices[3 * vi + 1],
                    attrib.vertices[3 * vi + 2]
                };
                if (ni != -1) {
                    vertex.normal = glm::vec3{ 
                        attrib.normals[3 * ni + 0],
                        attrib.normals[3 * ni + 1],
                        attrib.normals[3 * ni + 2]
                    };
                } else {
                    // to be calculated later
                    infer_normal = true;
                }
                
                if (uvi != -1) {
                    vertex.uv = glm::vec2{
                        attrib.texcoords[2 * uvi + 0],
                        attrib.texcoords[2 * uvi + 1]
                    };
                }


                vertices.emplace_back(vertex);
            }
        }
    }

    if(infer_normal) {
        // clear all normals
        for (auto& v : vertices) {
            v.normal = glm::vec3{ 0 };
        }
        for (size_t i = 2; i < vertices.size(); i += 3) {
            glm::vec3 normal = glm::cross(
                vertices[i - 1].position - vertices[i - 2].position,
                vertices[i - 0].position - vertices[i - 2].position
            );
            vertices[i - 2].normal += normal;
            vertices[i - 1].normal += normal;
            vertices[i].normal += normal;
        }
        for (auto& v : vertices) {
            v.normal = glm::normalize(v.normal);
        }
    }

    Object ret{ scene, Transform{}, std::move(mat), std::move(vertices) };
    return ret;
}

auto Object::make_triangle_obj(Scene const& scene, Material mat, Transform const& trans) noexcept -> Object {
	std::vector<Vertex> vertices = {
	    Vertex{ glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
	    Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
	    Vertex{ glm::vec3{ 0, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0.5, 1 } }
    };
    return Object { scene, trans, std::move(mat), std::move(vertices) };
}

auto Object::make_quad_obj(Scene const& scene, Material mat, Transform const& trans) noexcept -> Object {
    std::vector<Vertex> vertices = {
        Vertex{ glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },

        Vertex{ glm::vec3{ -0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 1 } }
    };
    return Object { scene, trans, std::move(mat), std::move(vertices) };
}

void Object::set_transform(Transform const& transform) noexcept {
    m_transform = transform;
    m_bound.min_pos = m_transform.local_to_world_pos(m_bound.min_pos);
    m_bound.max_pos = m_transform.local_to_world_pos(m_bound.max_pos);
}

auto Object::get_transform() const noexcept -> Transform const& {
    return m_transform;
}

void Object::set_name(std::string_view name) noexcept {
    m_name = name;
}

auto Object::get_name() const noexcept -> std::string_view {
    return m_name;
}

auto Object::get_bound() const noexcept -> BoundingBox const& {
    return m_bound;
}

auto Object::get_vertices() const noexcept -> std::vector<Vertex> const& {
    return m_vertices;
}

auto Object::get_material() const noexcept -> Material const& {
    return m_mat;
}
