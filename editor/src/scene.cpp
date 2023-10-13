#include "include/scene.h"
#include "include/ray.h"
#include "include/intersection.h"

Scene::Scene() = default;

auto Scene::from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string> {
    Scene ret{  };
    auto ores = Object::from_obj(ret, Material{}, filename);
    if (!ores) {
        return tl::unexpected{ ores.error() };
    }

    ret.m_objects.emplace_back(ores.value());
    return ret;
}

// here we assume +y is up
auto Scene::get_good_cam_start() const noexcept -> Transform {
    constexpr float tan_alpha = 0.4663f;

    if (m_objects.empty()) {
        return Transform::look_at(glm::vec3{ 0, 2, 2 }, glm::vec3{ 0 }, glm::vec3{ 0,1,0 });
    }

    auto const bound = compute_scene_bound();
    auto const center = bound.get_center();
    auto const extent = bound.get_extent();

    // local means relative to the bounding box
	float const cam_y_local = extent.y + 2;
    // tan(view_angle) = y_local / (extent.z + z_local)

    return Transform::look_at(
	    glm::vec3(center.x, center.y + cam_y_local, cam_y_local / tan_alpha),
        center,
        glm::vec3(0, 1, 0)
    );
}
// here we assume +y is up
auto Scene::get_good_light_pos() const noexcept -> glm::vec3 {
    auto const bound = compute_scene_bound();
    auto const center = bound.get_center();
    float const y_extent = bound.get_extent().y;
    return center + glm::vec3{ 0, y_extent + 3, 0 };
}

auto Scene::ray_cast(Ray const& ray, float t_min, float t_max) noexcept -> Object* {
    Object* ret = nullptr;
    float closest_t = t_max;
    for (auto&& obj : m_objects) {
        auto res = Intersection::ray_box(obj.bound, ray);
        if (res.hit && res.t < closest_t && res.t >= t_min) {
            closest_t = res.t;
            ret = &obj;
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

auto Scene::compute_scene_bound() const noexcept -> BoundingBox {
    BoundingBox scene_bound{
        glm::vec3 { std::numeric_limits<float>::max() },
        glm::vec3 { std::numeric_limits<float>::lowest() }
    };
    for (auto&& obj : m_objects) {
        scene_bound += obj.bound;
    }
    return scene_bound;
}
