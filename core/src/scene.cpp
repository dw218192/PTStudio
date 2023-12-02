#include "scene.h"
#include "ray.h"
#include "intersection.h"
#include "utils.h"
#include "boundingBox.h"
#include "boundingSphere.h"

#include <glm/ext/matrix_transform.hpp>
#include <algorithm>

auto PTS::Scene::from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string> {
    auto ret = Scene{};
    auto obj = RenderableObject{};
    TL_TRY_ASSIGN(obj, RenderableObject::from_obj(ret, Material{}, filename));
    ret.add_object(std::move(obj));

    return ret;
}

// here we assume +y is up
auto PTS::Scene::get_good_cam_start() const noexcept -> LookAtParams {
    constexpr float tan_alpha = 0.4663f;

    if (m_objects.empty()) {
        return { glm::vec3{ 0, 2, 2 }, glm::vec3{ 0 }, glm::vec3{ 0,1,0 } };
    }
    auto const bound = get_scene_bound();
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
auto PTS::Scene::get_good_light_pos() const noexcept -> glm::vec3 {
    if (m_objects.empty()) {
        return glm::vec3{ 0, 2, 2 };
    }
    auto const bound = get_scene_bound();
    auto const center = bound.get_center();
    auto const y_extent = bound.get_extent().y;
    return center + glm::vec3{ 0, y_extent + 3, 0 };
}

auto PTS::Scene::get_scene_bound() const noexcept -> BoundingBox {
    if (empty()) {
        return BoundingBox{ glm::vec3{ 0 }, glm::vec3{ 0 } };
    }
    auto ret = BoundingBox{};
    for (auto const& obj : m_objects) {
        ret += obj.get_bound();
    }
    return ret;
}

auto PTS::Scene::ray_cast_editable(Ray const& ray, float t_min, float t_max) noexcept -> ObserverPtr<SceneObject> {
    auto ret = ray_cast(ray, t_min, t_max);
    if (ret) {
        return ret;
    }

    // try select lights
	// intersect with gizmos for editing
    auto closest_t = t_max;
    auto const light_bound = BoundingSphere{ };
    for (auto&& light : m_lights) {
        if ((light.get_edit_flags() & EditFlags::_NoEdit) || 
            !(light.get_edit_flags() & EditFlags::Selectable)) {
            continue;
        }

        Ray local_ray{
            light.get_transform().world_to_local_pos(ray.origin),
            light.get_transform().world_to_local_dir(ray.direction)
        };
        auto const res = Intersection::ray_sphere(light_bound, local_ray);
        if (res.hit && res.t < closest_t && res.t >= t_min) {
            closest_t = res.t;
            ret = &light;
        }
    }
    return ret;
}

auto PTS::Scene::ray_cast(Ray const& ray, float t_min, float t_max) noexcept -> ObserverPtr<SceneObject> {
    auto ret = ObserverPtr<SceneObject> { nullptr };
    auto closest_t = t_max;
    for (auto&& obj : m_objects) {
        if ((obj.get_edit_flags() & EditFlags::_NoEdit) || 
            !(obj.get_edit_flags() & EditFlags::Selectable)) {
            continue;
        }

        // transform ray to object space
        Ray local_ray {
            obj.get_transform().world_to_local_pos(ray.origin),
            obj.get_transform().world_to_local_dir(ray.direction)
        };
        auto res = Intersection::ray_box(obj.get_bound(), local_ray);
        if (res.hit && res.t < closest_t && res.t >= t_min) {
            auto const& indices = obj.get_indices();
            auto const& verts = obj.get_vertices();
            for (size_t i=0; i < indices.size(); i+=3) {
                glm::vec3 triangle[3] {
                    verts[indices[i]].position,
                    verts[indices[i + 1]].position,
                    verts[indices[i + 2]].position
                };
            	res = Intersection::ray_triangle(triangle, local_ray);
                if (res.hit && res.t < closest_t && res.t >= t_min) {
                    closest_t = res.t;
                    ret = &obj;
                    break;
                }
            }
        }
    }

    return ret;
}

auto PTS::Scene::add_object(RenderableObject&& obj) noexcept -> ObserverPtr<RenderableObject> {
    auto const ret = &(m_objects.emplace_back(std::move(obj)));
    add_editable(*ret);
    return ret;
}

void PTS::Scene::remove_object(View<RenderableObject> obj_view) noexcept {
    auto&& obj = obj_view.get();
    remove_editable(obj);
    m_objects.remove_if([&](auto&& o) { return &o == &obj; });
}

auto PTS::Scene::add_light(Light&& light) noexcept -> ObserverPtr<Light> {
    auto const ret = &(m_lights.emplace_back(std::move(light)));
    add_editable(*ret);
    return ret;
}

void PTS::Scene::remove_light(View<Light> light_view) noexcept {
    auto&& light = light_view.get();
    remove_editable(light);

    for (auto [idx, l] = std::tuple{ 0, m_lights.begin() }; l != m_lights.end(); ++l, ++idx) {
        if (&(*l) == &light) {
            m_lights.erase(l);
            break;
        }
    }
}

void PTS::Scene::on_deserialize() noexcept {
    for (auto&& light : m_lights) {
        add_editable(light);
    }
    for (auto&& obj : m_objects) {
        add_editable(obj);
    }
}

auto PTS::Scene::add_editable(Ref<SceneObject> obj_view) noexcept -> void {
    if (!(obj_view.get().get_edit_flags() & EditFlags::_NoEdit)) {
        m_editables.emplace_back(obj_view);
    }
}

auto PTS::Scene::remove_editable(View<SceneObject> obj_view) noexcept -> void {
    if (!(obj_view.get().get_edit_flags() & EditFlags::_NoEdit)) {
        m_editables.remove_if([&](auto&& obj) { return &obj.get() == &obj_view.get(); });
    }
}
