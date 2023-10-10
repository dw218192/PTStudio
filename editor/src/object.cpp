#include "include/object.h"
#include "include/camera.h"

#include <iostream>

Object::Object() = default;
Object::Object(Material mat) : m_bound{}, m_material{mat} { }

auto Object::from_obj(Material mat, std::string_view filename) noexcept -> tl::expected<Object, std::string> {
    Object ret{ mat };
	ret.m_vertices.clear();

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
    glm::vec3 min_pos{
        std::numeric_limits<float>::max()
    };
    glm::vec3 max_pos{
        std::numeric_limits<float>::lowest()
    };

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


                ret.m_vertices.emplace_back(vertex);
                min_pos = glm::min(vertex.position, min_pos);
                max_pos = glm::max(vertex.position, max_pos);
            }
        }
    }

    if(infer_normal) {
        // clear all normals
        for (auto& v : ret.m_vertices) {
            v.normal = glm::vec3{ 0 };
        }
        for (size_t i = 2; i < ret.m_vertices.size(); i += 3) {
            glm::vec3 normal = glm::cross(
                ret.m_vertices[i - 1].position - ret.m_vertices[i - 2].position,
                ret.m_vertices[i - 0].position - ret.m_vertices[i - 2].position
            );
            ret.m_vertices[i - 2].normal += normal;
            ret.m_vertices[i - 1].normal += normal;
            ret.m_vertices[i].normal += normal;
        }
        for (auto& v : ret.m_vertices) {
            v.normal = glm::normalize(v.normal);
        }
    }

    ret.m_bound = { min_pos, max_pos };
    return ret;
}

auto Object::make_triangle_obj(Material mat, Transform const& trans) noexcept -> Object {
    Object obj{ mat };
    obj.m_transform = trans;

	obj.m_vertices = {
	    Vertex{ glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
	    Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
	    Vertex{ glm::vec3{ 0, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0.5, 1 } }
    };
    obj.m_bound = { glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0.5, 0.5, 0 } };
    
    return obj;
}

auto Object::make_quad_obj(Material mat, Transform const& trans) noexcept -> Object {
    Object obj{ mat };
    obj.m_transform = trans;

    obj.m_vertices = {
        Vertex{ glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },

        Vertex{ glm::vec3{ -0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 1 } }
    };
    obj.m_bound = { glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0.5, 0.5, 0 } };

    return obj;
}
