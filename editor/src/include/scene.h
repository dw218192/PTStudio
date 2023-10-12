#pragma once
#include "object.h"
#include <tl/expected.hpp>
#include <string_view>
#include <vector>
#include <optional>

#include "camera.h"

struct Scene {
    Scene();

    /**
     * \brief Creates a scene from an obj file.
     * \param filename the path to the obj file
     * \return nothing if the file was loaded successfully, an error otherwise
    */
    [[nodiscard]] static auto from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string>;
    // for test only
    [[nodiscard]] static auto make_triangle_scene() noexcept -> tl::expected<Scene, std::string>;

    [[nodiscard]] auto get_objects() const noexcept -> std::vector<Object> const& { return m_objects; }

	// compute good positions to place light and camera
	[[nodiscard]] auto get_good_cam_start() const noexcept -> Transform;
    [[nodiscard]] auto get_good_light_pos() const noexcept -> glm::vec3;

    [[nodiscard]] auto ray_cast(struct Ray const& ray, float t_min = 0.0f, float t_max = 1e5f) const noexcept -> std::optional<Object>;

private:
    [[nodiscard]] auto compute_scene_bound() const noexcept -> BoundingBox;

    // for now all objects use this shader
    static inline ShaderProgram s_default_shader;
    // these are initialized when the scene is loaded
	std::vector<Object> m_objects;
};
