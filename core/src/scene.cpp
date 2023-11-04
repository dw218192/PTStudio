#include "scene.h"
#include "ray.h"
#include "intersection.h"

#include <glm/ext/matrix_transform.hpp>
#include <algorithm>

Scene::Scene() = default;

auto Scene::from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string> {
    Scene ret{  };
    auto ores = Object::from_obj(ret, Material{}, filename);
    if (!ores) {
        return TL_ERROR( ores.error() );
    }

    ret.m_objects.emplace_back(ores.value());
    return ret;
}

// here we assume +y is up
auto Scene::get_good_cam_start() const noexcept -> LookAtParams {
    constexpr float tan_alpha = 0.4663f;

    if (m_objects.empty()) {
        return { glm::vec3{ 0, 2, 2 }, glm::vec3{ 0 }, glm::vec3{ 0,1,0 } };
    }

    auto const bound = compute_scene_bound();
    auto const center = bound.get_center();
    auto const extent = bound.get_extent();

    // local means relative to the bounding box
	float const cam_y_local = extent.y + 2;
    // tan(view_angle) = y_local / (extent.z + z_local)

    return {
        glm::vec3(center.x, center.y + cam_y_local, cam_y_local / tan_alpha),
        center,
        glm::vec3(0, 1, 0)
    };
}
// here we assume +y is up
auto Scene::get_good_light_pos() const noexcept -> glm::vec3 {
    auto const bound = compute_scene_bound();
    auto const center = bound.get_center();
    float const y_extent = bound.get_extent().y;
    return center + glm::vec3{ 0, y_extent + 3, 0 };
}

auto Scene::ray_cast_editable(Ray const& ray, float t_min, float t_max) noexcept -> std::optional<EditableView> {
    auto ret = ray_cast(ray, t_min, t_max);
    if (ret) {
        return ret;
    }

    // try select lights
	// intersect with gizmos for editing
    float closest_t = t_max;
    auto const light_bound = BoundingBox{ glm::vec3{ -0.5f }, glm::vec3{ 0.5f } };
    for (auto&& light : m_lights) {
        Ray local_ray{
            light.get_transform().world_to_local_pos(ray.origin),
            light.get_transform().world_to_local_dir(ray.direction)
        };
        auto const res = Intersection::ray_box(light_bound, local_ray);
        if (res.hit && res.t < closest_t && res.t >= t_min) {
            closest_t = res.t;
            ret = light;
        }
    }
    return ret;
}

auto Scene::ray_cast(Ray const& ray, float t_min, float t_max) noexcept -> std::optional<EditableView> {
    std::optional<EditableView> ret = std::nullopt;
    float closest_t = t_max;
    for (auto&& obj : m_objects) {
        // transform ray to object space
        Ray local_ray {
            obj.get_transform().world_to_local_pos(ray.origin),
            obj.get_transform().world_to_local_dir(ray.direction)
        };
        auto res = Intersection::ray_box(obj.get_bound(), local_ray);
        if (res.hit && res.t < closest_t && res.t >= t_min) {
            auto&& triangles = obj.get_vertices();
            for (int i=0; i<triangles.size(); i+=3) {
                auto triangle = {
                    triangles[i].position,
                    triangles[i + 1].position,
                    triangles[i + 2].position
                };

            	res = Intersection::ray_triangle(tcb::make_span(triangle), local_ray);
                if (res.hit && res.t < closest_t && res.t >= t_min) {
                    closest_t = res.t;
                    ret = obj;
                    break;
                }
            }
        }
    }

    return ret;
}

auto Scene::make_triangle_scene() noexcept -> tl::expected<Scene, std::string> {
    Scene scene { };
    scene.m_objects.emplace_back(Object::make_triangle_obj(scene, Material{},
        Transform{  }));
    scene.m_objects.emplace_back(Object::make_triangle_obj(scene, Material{},
        Transform{ glm::vec3{0.2,0.2,0}, glm::vec3{0,0,0}, glm::vec3{1,1,1} }));
    return scene;
}

auto Scene::add_object(Object obj) noexcept -> ObserverPtr<Object> {
    auto const ret = &(m_objects.emplace_back(std::move(obj)));
    m_editables.emplace_back(*ret);
    return ret;
}

void Scene::remove_object(View<Object> obj_view) noexcept {
    auto&& obj = obj_view.get();
    remove_editable(obj);
    m_objects.remove_if([&](auto&& o) { return &o == &obj; });
}

auto Scene::add_light(Light light) noexcept -> ObserverPtr<Light> {
    auto const ret = &(m_lights.emplace_back(std::move(light)));
    m_editables.emplace_back(*ret);
    return ret;
}

void Scene::remove_light(View<Light> light_view) noexcept {
    auto&& light = light_view.get();
    remove_editable(light);

    for (auto [idx, l] = std::tuple{ 0, m_lights.begin() }; l != m_lights.end(); ++l, ++idx) {
        if (&(*l) == &light) {
            m_lights.erase(l);
            break;
        }
    }
}

void Scene::on_deserialize() noexcept {
    for (auto&& light : m_lights) {
        m_editables.emplace_back(light);
    }

    for (auto&& obj : m_objects) {
        m_editables.emplace_back(obj);
    }
}

auto Scene::compute_scene_bound() const noexcept -> BoundingBox {
    BoundingBox scene_bound{
        glm::vec3 { std::numeric_limits<float>::max() },
        glm::vec3 { std::numeric_limits<float>::lowest() }
    };
    for (auto&& obj : m_objects) {
        scene_bound += obj.get_bound();
    }
    return scene_bound;
}

auto Scene::remove_editable(ConstEditableView editable) noexcept -> void {
    m_editables.remove_if([&](auto&& obj) { return obj == editable; });
}
