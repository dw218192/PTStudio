#include "include/object.h"
#include "include/camera.h"

#include <iostream>

Object::Object(ShaderProgram const& shader, Transform transform) 
    : m_transform{std::move(transform)}, m_program{&shader} 
{ }

auto Object::from_obj(std::string_view filename) noexcept -> Result<void> {
    m_vertices.clear();

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "./";

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename.data(), config)) {
        if (!reader.Error().empty()) {
            return "Tiny Obj Error:\n" + reader.Error();
        }
        else {
            return std::string{ "Unknown Tiny Obj Error" };
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


                m_vertices.emplace_back(vertex);
                min_pos = glm::min(vertex.position, min_pos);
                max_pos = glm::max(vertex.position, max_pos);
            }
        }
    }

    if(infer_normal) {
        // clear all normals
        for (auto& v : m_vertices) {
            v.normal = glm::vec3{ 0 };
        }
        for (size_t i = 2; i < m_vertices.size(); i += 3) {
            glm::vec3 normal = glm::cross(
                m_vertices[i - 1].position - m_vertices[i - 2].position,
                m_vertices[i - 0].position - m_vertices[i - 2].position
            );
            m_vertices[i - 2].normal += normal;
            m_vertices[i - 1].normal += normal;
            m_vertices[i].normal += normal;
        }
        for (auto& v : m_vertices) {
            v.normal = glm::normalize(v.normal);
        }
    }

    m_bound = { min_pos, max_pos };
    return Result<void>::ok();
}

// TODO: if we want to support multi-shader, this should be moved to the ShaderProgram class
auto Object::begin_draw(Camera const& cam) const noexcept -> Result<void> {
    m_program->use();
    auto res = m_program->set_uniform(k_uniform_model, m_transform.get_matrix());
    if (!res.valid()) {
        return res;
    }
    res = m_program->set_uniform(k_uniform_view, cam.get_transform().get_matrix());
    if (!res.valid()) {
        return res;
    }
    res = m_program->set_uniform(k_uniform_projection, cam.get_projection());
    if (!res.valid()) {
        return res;
    }

    return Result<void>::ok();
}

void Object::end_draw() const noexcept {
    m_program->unuse();
}

auto Object::make_triangle_obj(ShaderProgram const& shader, Transform const& trans) noexcept -> Object {
    Object obj{ shader, trans };
	obj.m_vertices = {
	    Vertex{ glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
	    Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
	    Vertex{ glm::vec3{ 0, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0.5, 1 } }
    };
    obj.m_bound = { glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0.5, 0.5, 0 } };
    
    return obj;
}
