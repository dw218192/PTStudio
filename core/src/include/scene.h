#pragma once


#include <string>
#include <tl/expected.hpp>
#include <tcb/span.hpp>
#include <vector>

#include "camera.h"
#include "boundingBox.h"
#include "utils.h"
#include "reflection.h"
#include "object.h"
#include "light.h"
#include "editableView.h"

struct Scene {
    Scene();

    /**
     * \brief Creates a scene from an obj file.
     * \param filename the path to the obj file
     * \return nothing if the file was loaded successfully, an error otherwise
    */
    NODISCARD static auto from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string>;

	// compute good positions to place light and camera
	NODISCARD auto get_good_cam_start() const noexcept -> LookAtParams;
    NODISCARD auto get_good_light_pos() const noexcept -> glm::vec3;
    NODISCARD auto get_scene_bound() const noexcept {
        if(empty()) {
            return BoundingBox{ glm::vec3{ 0 }, glm::vec3{ 0 } };
        }
        return m_scene_bound;
    }

    NODISCARD auto ray_cast(Ray const& ray, float t_min = 0.0f, float t_max = 1e5f) noexcept -> std::optional<EditableView>;
    NODISCARD auto ray_cast_editable(Ray const& ray, float t_min = 0.0f, float t_max = 1e5f) noexcept -> std::optional<EditableView>;
    NODISCARD auto size() const noexcept { return m_objects.size(); }
    NODISCARD auto empty() const noexcept -> bool { return m_objects.empty(); }

    auto add_object(Object obj) noexcept -> ObserverPtr<Object>;
    void remove_object(View<Object> obj_view) noexcept;

    auto add_light(Light light) noexcept -> ObserverPtr<Light>;
    void remove_light(View<Light> light_view) noexcept;

    NODISCARD auto get_objects() const noexcept -> auto const& { return m_objects; }
    NODISCARD auto get_lights() const noexcept -> auto const& { return m_lights; }
    NODISCARD auto get_editables() const noexcept -> auto const& { return m_editables; }

    NODISCARD auto next_obj_name() const noexcept -> std::string {
        static int counter = 0;
        return "Object " + std::to_string(counter++);
    }
    NODISCARD auto next_light_name() const noexcept -> std::string {
        static int counter = 0;
        return "Light " + std::to_string(counter++);
    }
    NODISCARD auto get_name() const noexcept -> std::string_view { return m_name; }
    auto set_name(std::string_view name) noexcept -> void { m_name = name; }

    void on_deserialize() noexcept;
private:
    auto remove_editable(ConstEditableView editable) noexcept -> void;

    BEGIN_REFLECT(Scene);
	    FIELD_MOD(std::string, m_name, "Scene",
            MSerialize{});

		FIELD_MOD(std::list<Object>, m_objects, {},
            MSerialize{});

        FIELD_MOD(std::list<Light>, m_lights, {},
            MSerialize{});
        
        FIELD_MOD(BoundingBox, m_scene_bound, {},
            MSerialize{});
    END_REFLECT();

private:
    std::list<EditableView> m_editables;
};
