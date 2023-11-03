#pragma once


#include <string>
#include <tl/expected.hpp>
#include <vector>

#include "camera.h"
#include "boundingBox.h"
#include "utils.h"
#include "reflection.h"
#include "object.h"
#include "light.h"

struct Scene {
    Scene();

    /**
     * \brief Creates a scene from an obj file.
     * \param filename the path to the obj file
     * \return nothing if the file was loaded successfully, an error otherwise
    */
    NODISCARD static auto from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string>;
    // for test only
    NODISCARD static auto make_triangle_scene() noexcept -> tl::expected<Scene, std::string>;

	// compute good positions to place light and camera
	NODISCARD auto get_good_cam_start() const noexcept -> LookAtParams;
    NODISCARD auto get_good_light_pos() const noexcept -> glm::vec3;

    NODISCARD auto ray_cast(Ray const& ray, float t_min = 0.0f, float t_max = 1e5f) noexcept -> ObserverPtr<Object>;
    NODISCARD auto size() const noexcept { return m_objects.size(); }

    auto add_object(Object obj) noexcept -> ObserverPtr<Object>;
    void remove_object(ObserverPtr<Object> obj) noexcept;

    auto add_light(Light light) noexcept -> ObserverPtr<Light>;
    void remove_light(ObserverPtr<Light> light) noexcept;

    auto get_objects() const noexcept -> std::vector<Object> const& { return m_objects; }
    auto get_lights() const noexcept -> std::vector<Light> const& { return m_lights; }
    
    NODISCARD auto next_obj_name() const noexcept -> std::string {
        static int counter = 0;
        return "Object " + std::to_string(counter++);
    }

    NODISCARD auto get_name() const noexcept -> std::string_view { return m_name; }
private:
    NODISCARD auto compute_scene_bound() const noexcept -> BoundingBox;

    BEGIN_REFLECT(Scene);
	    FIELD_MOD(std::string, m_name, 
            MDefault{ "Scene" },
            MSerialize{});

		FIELD_MOD(std::vector<Object>, m_objects,
            MSerialize{});

        FIELD_MOD(std::vector<Light>, m_lights,
            MSerialize{});
    END_REFLECT();
};
