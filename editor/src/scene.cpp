#include "include/scene.h"

auto Scene::from_obj_file(std::string_view filename) noexcept -> Result<void> {
    auto const res = get_default_shader();
    if(!res.valid()) {
        return res.error();
    }

    Object obj{ res.value(), Transform{} };
    if(auto ores = obj.from_obj(filename); !ores.valid()) {
        return ores;
    }
    m_objects.emplace_back(std::move(obj));
    return Result<void>::ok();
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

auto Scene::begin_draw() const noexcept -> Result<void> {
    s_default_shader.use();
    auto const res = s_default_shader.set_uniform(k_uniform_light_pos, get_good_light_pos());
    if (!res.valid()) {
        return res.error();
    }

    return Result<void>::ok();
}

auto Scene::make_triangle_scene() noexcept -> Result<Scene> {
    auto const res = get_default_shader();
    if (!res.valid()) {
        return res.error();
    }
    Scene scene;
    scene.m_objects.emplace_back(Object::make_triangle_obj(res.value(), 
        Transform{  }));
    scene.m_objects.emplace_back(Object::make_triangle_obj(res.value(), 
        Transform{ glm::vec3{0.2,0.2,0}, glm::vec3{0,0,0}, glm::vec3{1,1,1} }));
    return scene;
}

auto Scene::get_default_shader() noexcept -> Result<ShaderProgram const&> {
    if (!s_default_shader.valid()) {
        // initialize default shader program
        Shader vs(ShaderType::Vertex);
        {
            auto const res = vs.from_src(vs_obj_src);
            if (!res.valid()) {
                return res.error();
            }
        }
        Shader fs(ShaderType::Fragment);
        {
            auto const res = fs.from_src(ps_obj_src);
            if (!res.valid()) {
                return res.error();
            }
        }
        {
            auto const res = s_default_shader.from_shaders(std::move(vs), std::move(fs));
            if (!res.valid()) {
                return res.error();
            }
        }
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
