#include <core/legacy/boundingBox.h>
#include <core/legacy/boundingSphere.h>
#include <core/legacy/intersection.h>
#include <core/legacy/ray.h>
#include <core/legacy/scene.h>
#include <core/legacy/utils.h>

#include <algorithm>
#include <glm/ext/matrix_transform.hpp>

// here we assume +y is up
auto PTS::Scene::get_good_cam_start() const noexcept -> LookAtParams {
    constexpr auto tan_alpha = 0.4663f;

    if (m_renderable_objects.empty()) {
        return {glm::vec3{0, 2, 2}, glm::vec3{0}, glm::vec3{0, 1, 0}};
    }
    auto const bound = get_scene_bound();
    auto const center = bound.get_center();
    auto const extent = bound.get_extent();

    // local means relative to the bounding box
    auto const cam_y_local = extent.y + 2;
    // tan(view_angle) = y_local / (extent.z + z_local)

    return {glm::vec3(center.x, center.y + cam_y_local, cam_y_local / tan_alpha), center,
            glm::vec3(0, 1, 0)};
}

// here we assume +y is up
auto PTS::Scene::get_good_light_pos() const noexcept -> glm::vec3 {
    if (m_renderable_objects.empty()) {
        return glm::vec3{0, 2, 2};
    }
    auto const bound = get_scene_bound();
    auto const center = bound.get_center();
    auto const y_extent = bound.get_extent().y;
    return center + glm::vec3{0, y_extent + 3, 0};
}

auto PTS::Scene::get_scene_bound() const noexcept -> BoundingBox {
    if (empty()) {
        return BoundingBox{glm::vec3{0}, glm::vec3{0}};
    }
    auto ret = BoundingBox{};
    for (auto const& obj : m_renderable_objects) {
        ret += obj.get_bound();
    }
    return ret;
}

auto PTS::Scene::ray_cast_editable(Ray const& ray, float t_min,
                                   float t_max) noexcept -> ObserverPtr<SceneObject> {
    auto ret = ray_cast(ray, t_min, t_max);
    if (ret) {
        return ret;
    }

    // try select lights
    // intersect with gizmos for editing
    auto closest_t = t_max;
    auto const light_bound = BoundingSphere{};
    for (auto&& light : m_lights) {
        if (!light.is_editable() || !(light.get_edit_flags() & EditFlags::Selectable)) {
            continue;
        }

        Ray local_ray{light.get_transform(TransformSpace::WORLD).world_to_local_pos(ray.origin),
                      light.get_transform(TransformSpace::WORLD).world_to_local_dir(ray.direction)};
        auto const res = Intersection::ray_sphere(light_bound, local_ray);
        if (res.hit && res.t < closest_t && res.t >= t_min) {
            closest_t = res.t;
            ret = &light;
        }
    }
    return ret;
}

auto PTS::Scene::ray_cast(Ray const& ray, float t_min,
                          float t_max) noexcept -> ObserverPtr<SceneObject> {
    auto ret = ObserverPtr<SceneObject>{nullptr};
    auto closest_t = t_max;
    for (auto&& obj : m_renderable_objects) {
        if (!obj.is_editable() || !(obj.get_edit_flags() & EditFlags::Selectable)) {
            continue;
        }

        // transform ray to object space
        Ray local_ray{obj.get_transform(TransformSpace::WORLD).world_to_local_pos(ray.origin),
                      obj.get_transform(TransformSpace::WORLD).world_to_local_dir(ray.direction)};
        auto res = Intersection::ray_box(obj.get_bound(), local_ray);
        if (res.hit && res.t < closest_t && res.t >= t_min) {
            auto const& indices = obj.get_indices();
            auto const& verts = obj.get_vertices();
            for (size_t i = 0; i < indices.size(); i += 3) {
                glm::vec3 triangle[3]{verts[indices[i]].position, verts[indices[i + 1]].position,
                                      verts[indices[i + 2]].position};
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

auto PTS::Scene::is_valid_obj(View<SceneObject> obj_view) const noexcept -> bool {
    return m_alive_objs.count(&obj_view.get());
}

void PTS::Scene::on_deserialize() noexcept {
    for (auto&& light : m_lights) {
        try_add_editable(light);
    }
    for (auto&& obj : m_renderable_objects) {
        try_add_editable(obj);
    }
}

auto PTS::Scene::try_add_editable(Ref<SceneObject> obj_view) noexcept -> void {
    if (obj_view.get().is_editable()) {
        m_editables.emplace_back(obj_view);
    }
}

auto PTS::Scene::try_remove_editable(View<SceneObject> obj_view) noexcept -> void {
    m_editables.remove_if([&](auto&& obj) { return &obj.get() == &obj_view.get(); });
}

auto PTS::Scene::get_callback_list(SceneChangeType type) -> CallbackList<void(Ref<SceneObject>)>& {
    return m_scene_callbacks[type];
}
