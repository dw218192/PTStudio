#include "object.h"
#include "camera.h"
#include "scene.h"

#include <tiny_obj_loader.h>
#include <glm/ext/scalar_constants.hpp>

Object::Object(Scene const& scene, Transform transform, Material mat, tcb::span<Vertex const> vertices, std::string_view name)
    : m_mat { std::move(mat) }, m_vertices { vertices.begin(), vertices.end() }, m_name { name }
{
    m_local_bound = BoundingBox::from_vertices(m_vertices);
    set_transform(transform);
}

Object::Object(Scene const& scene, Transform transform, Material mat, tcb::span<Vertex const> vertices)
    : m_mat { std::move(mat) }, m_vertices{ vertices.begin(), vertices.end() }, m_name { scene.next_obj_name() }
{
    m_local_bound = BoundingBox::from_vertices(m_vertices);
    set_transform(transform);
}

Object::Object(Transform transform, Material mat, tcb::span<Vertex const> vertices, std::string_view name) 
    : m_mat { std::move(mat) }, m_vertices{ vertices.begin(), vertices.end() }, m_name { name }
{
    m_local_bound = BoundingBox::from_vertices(m_vertices);
    set_transform(transform);
}

Object::Object(Transform transform, Material mat, tcb::span<Vertex const> vertices)
    : m_mat { std::move(mat) }, m_vertices{ vertices.begin(), vertices.end() }, m_name { "Object" }
{
    m_local_bound = BoundingBox::from_vertices(m_vertices);
    set_transform(transform);
}

auto Object::from_obj(Scene const& scene, Material mat, std::string_view filename, std::string* warning) noexcept 
-> tl::expected<Object, std::string> {
    std::vector<Vertex> vertices;

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "./";

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename.data(), config)) {
        if (!reader.Error().empty()) {
            return TL_ERROR( "Tiny Obj Error:\n" + reader.Error() );
        }
        else {
            return TL_ERROR("Unknown Tiny Obj Error" );
        }
    }

    if (!reader.Warning().empty() && warning) {
        *warning = reader.Warning();
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

auto Object::make_triangle_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object {
	std::vector<Vertex> vertices = {
	    Vertex{ glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
	    Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
	    Vertex{ glm::vec3{ 0, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0.5, 1 } }
    };
    return Object { scene, std::move(trans), std::move(mat), std::move(vertices) };
}

auto Object::make_quad_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object {
    std::vector<Vertex> vertices = {
        Vertex{ glm::vec3{ -0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },

        Vertex{ glm::vec3{ -0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, 0 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 1 } }
    };
    return Object { scene, std::move(trans), std::move(mat), std::move(vertices) };
}

auto Object::make_cube_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object {
    std::vector<Vertex> vertices = {
        // front
        Vertex{ glm::vec3{ -0.5, -0.5, 0.5 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0.5 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, 0.5 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },

        Vertex{ glm::vec3{ -0.5, 0.5, 0.5 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0.5 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, 0.5 }, glm::vec3{ 0, 0, 1 }, glm::vec2{ 1, 1 } },

        // back
        Vertex{ glm::vec3{ -0.5, -0.5, -0.5 }, glm::vec3{ 0, 0, -1 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, -0.5 }, glm::vec3{ 0, 0, -1 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, -0.5 }, glm::vec3{ 0, 0, -1 }, glm::vec2{ 1, 0 } },

        Vertex{ glm::vec3{ 0.5, -0.5, -0.5 }, glm::vec3{ 0, 0, -1 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, -0.5 }, glm::vec3{ 0, 0, -1 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, 0.5, -0.5 }, glm::vec3{ 0, 0, -1 }, glm::vec2{ 1, 1 } },

        // left
        Vertex{ glm::vec3{ -0.5, -0.5, -0.5 }, glm::vec3{ -1, 0, 0 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, -0.5 }, glm::vec3{ -1, 0, 0 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ -0.5, -0.5, 0.5 }, glm::vec3{ -1, 0, 0 }, glm::vec2{ 1, 0 } },

        Vertex{ glm::vec3{ -0.5, -0.5, 0.5 }, glm::vec3{ -1, 0, 0 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, -0.5 }, glm::vec3{ -1, 0, 0 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ -0.5, 0.5, 0.5 }, glm::vec3{ -1, 0, 0 }, glm::vec2{ 1, 1 } },

        // right
        Vertex{ glm::vec3{ 0.5, -0.5, -0.5 }, glm::vec3{ 1, 0, 0 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0.5 }, glm::vec3{ 1, 0, 0 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, -0.5 }, glm::vec3{ 1, 0, 0 }, glm::vec2{ 0, 1 } },

        Vertex{ glm::vec3{ 0.5, 0.5, -0.5 }, glm::vec3{ 1, 0, 0 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0.5 }, glm::vec3{ 1, 0, 0 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, 0.5 }, glm::vec3{ 1, 0, 0 }, glm::vec2{ 1, 1 } },

        // top
        Vertex{ glm::vec3{ -0.5, 0.5, -0.5 }, glm::vec3{ 0, 1, 0 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, -0.5 }, glm::vec3{ 0, 1, 0 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, 0.5, 0.5 }, glm::vec3{ 0, 1, 0 }, glm::vec2{ 0, 1 } },

        Vertex{ glm::vec3{ -0.5, 0.5, 0.5 }, glm::vec3{ 0, 1, 0 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, 0.5, -0.5 }, glm::vec3{ 0, 1, 0 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ 0.5, 0.5, 0.5 }, glm::vec3{ 0, 1, 0 }, glm::vec2{ 1, 1 } },

        // bottom
        Vertex{ glm::vec3{ -0.5, -0.5, -0.5 }, glm::vec3{ 0, -1, 0 }, glm::vec2{ 0, 0 } },
        Vertex{ glm::vec3{ -0.5, -0.5, 0.5 }, glm::vec3{ 0, -1, 0 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, -0.5 }, glm::vec3{ 0, -1, 0 }, glm::vec2{ 1, 0 } },

        Vertex{ glm::vec3{ 0.5, -0.5, -0.5 }, glm::vec3{ 0, -1, 0 }, glm::vec2{ 1, 0 } },
        Vertex{ glm::vec3{ -0.5, -0.5, 0.5 }, glm::vec3{ 0, -1, 0 }, glm::vec2{ 0, 1 } },
        Vertex{ glm::vec3{ 0.5, -0.5, 0.5 }, glm::vec3{ 0, -1, 0 }, glm::vec2{ 1, 1 } },
    };
    return Object { scene, std::move(trans), std::move(mat), std::move(vertices) };
}

auto Object::make_sphere_obj(Scene const& scene, Material mat, Transform trans) noexcept -> Object {
    // see http://www.songho.ca/opengl/gl_sphere.html
    std::vector<Vertex> points;

    // will make these customizable later as a component
    float radius = 1.0f;
    float length_inv = 1.0f / radius;
    int sector_count = 36;
    int stack_count = 18;
    float sector_step = 2 * glm::pi<float>() / sector_count;
    float stack_step = glm::pi<float>() / stack_count;
    for(int i = 0; i <= stack_count; ++i)
    {
        auto stack_angle = glm::pi<float>() / 2 - i * stack_step;
        auto xy = radius * glm::cos(stack_angle);
        auto z = radius * glm::sin(stack_angle);

        for(int j = 0; j <= sector_count; ++j) {
            auto sectorAngle = j * sector_step;       
            auto x = xy * cosf(sectorAngle);
            auto y = xy * sinf(sectorAngle);
            auto nx = x * length_inv;
            auto ny = y * length_inv;
            auto nz = z * length_inv;
            auto s = static_cast<float>(j) / sector_count;
            auto t = static_cast<float>(i) / sector_count;
            points.emplace_back(Vertex{ glm::vec3{ x, y, z }, glm::vec3{ nx, ny, nz }, glm::vec2{ s, t } });
        }
    }

    std::vector<Vertex> vertices;
    for(int i = 0; i < stack_count; ++i) {
        auto k1 = i * (sector_count + 1);
        auto k2 = k1 + sector_count + 1;
        for(int j = 0; j < sector_count; ++j, ++k1, ++k2) {
            if(i != 0) {
                vertices.emplace_back(points[k1]);
                vertices.emplace_back(points[k2]);
                vertices.emplace_back(points[k1 + 1]);
            }
            if(i != (stack_count - 1)) {
                vertices.emplace_back(points[k1 + 1]);
                vertices.emplace_back(points[k2]);
                vertices.emplace_back(points[k2 + 1]);
            }
        }
    }

    return Object { scene, std::move(trans), std::move(mat), std::move(vertices) };
}
