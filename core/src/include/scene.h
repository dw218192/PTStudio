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
#include "renderableObject.h"
#include "light.h"

namespace PTS {
    struct Scene : Object {
        Scene() : Object("Scene") {}

        /**
         * \brief Creates a scene from an obj file.
         * \param filename the path to the obj file
         * \return nothing if the file was loaded successfully, an error otherwise
        */
        NODISCARD static auto from_obj_file(std::string_view filename) noexcept -> tl::expected<Scene, std::string>;

        // compute good positions to place light and camera
        NODISCARD auto get_good_cam_start() const noexcept -> LookAtParams;
        NODISCARD auto get_good_light_pos() const noexcept -> glm::vec3;
        NODISCARD auto get_scene_bound() const noexcept -> BoundingBox;

        NODISCARD auto ray_cast(Ray const& ray, float t_min = 0.0f, float t_max = 1e5f) noexcept -> ObserverPtr<SceneObject>;
        NODISCARD auto ray_cast_editable(Ray const& ray, float t_min = 0.0f, float t_max = 1e5f) noexcept -> ObserverPtr<SceneObject>;
        NODISCARD auto size() const noexcept { return m_objects.size(); }
        NODISCARD auto empty() const noexcept -> bool { return m_objects.empty(); }

        auto add_object(RenderableObject&& obj) noexcept -> ObserverPtr<RenderableObject>;
        void remove_object(View<RenderableObject> obj_view) noexcept;
        auto add_light(Light&& light) noexcept -> ObserverPtr<Light>;
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

        void on_deserialize() noexcept;
        auto add_editable(Ref<SceneObject> obj_view) noexcept -> void;
        void remove_editable(View<SceneObject> obj_view) noexcept;
    private:
        BEGIN_REFLECT(Scene, Object);
        FIELD(std::list<RenderableObject>, m_objects, {},
            MSerialize{});
        FIELD(std::list<Light>, m_lights, {},
            MSerialize{});
        END_REFLECT();
        // enables dynamic retrieval of class info for polymorphic types
        DECL_DYNAMIC_INFO();
    private:
        std::list<Ref<SceneObject>> m_editables;
    };
}