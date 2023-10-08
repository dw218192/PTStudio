#include "include/scene.h"

auto Scene::from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string> {
    auto res = get_default_shader();
    if (!res) {
        return tl::unexpected{ res.error() };
    }
    auto ores = Object::from_obj(res.value(), filename);
    if(!ores) {
        return tl::unexpected{ ores.error() };
    }

    Scene ret;
    ret.m_objects.emplace_back(ores.value());
    return ret;
}

// here we assume +y is up
auto Scene::get_good_cam_start() const noexcept -> Transform {
    constexpr float tan_alpha = 0.4663f;

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

auto Scene::begin_draw() const noexcept -> tl::expected<void, std::string> {
    s_default_shader.use();
    return s_default_shader.set_uniform(k_uniform_light_pos, get_good_light_pos());
}

auto Scene::make_triangle_scene() noexcept -> tl::expected<Scene, std::string> {
    auto res = get_default_shader();
    if (!res) {
        return tl::unexpected{ res.error() };
    }
    Scene scene;
    scene.m_objects.emplace_back(Object::make_triangle_obj(res.value(), 
        Transform{  }));
    scene.m_objects.emplace_back(Object::make_triangle_obj(res.value(), 
        Transform{ glm::vec3{0.2,0.2,0}, glm::vec3{0,0,0}, glm::vec3{1,1,1} }));
    return scene;
}

auto Scene::get_default_shader() noexcept -> tl::expected<ShaderProgramRef, std::string> {
    if (!s_default_shader.valid()) {
        auto vs_res = Shader::from_src(ShaderType::Vertex, vs_obj_src);
        if (!vs_res) {
            return tl::unexpected{ vs_res.error() };
        }
        auto ps_res = Shader::from_src(ShaderType::Fragment, ps_obj_src);
        if (!ps_res) {
            return tl::unexpected{ ps_res.error() };
        }

        auto res = ShaderProgram::from_shaders(std::move(vs_res.value()), std::move(ps_res.value()));
        if (!res) {
            return tl::unexpected{ res.error() };
        }
        s_default_shader = std::move(res.value());
    }
    return s_default_shader;
}

AABB Scene::compute_scene_bound() const noexcept {
    AABB scene_bound{
	glm::vec3 { std::numeric_limits<float>::max() },
	glm::vec3 { std::numeric_limits<float>::lowest() }
    };
    for (auto&& obj : m_objects) {
        scene_bound.min_pos = glm::min(scene_bound.min_pos, obj.bound().min_pos);
        scene_bound.max_pos = glm::max(scene_bound.max_pos, obj.bound().max_pos);
    }
    return scene_bound;
}
