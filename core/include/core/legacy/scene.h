#pragma once

#include <list>
#include <string>
#include <tcb/span.hpp>
#include <tl/expected.hpp>

#include <core/signal.h>

#include "boundingBox.h"
#include "camera.h"
#include "enumArray.h"
#include "light.h"
#include "object.h"
#include "reflection.h"
#include "renderableObject.h"
#include "sceneObject.h"
#include "utils.h"

namespace PTS {
DECL_ENUM(SceneChangeType, OBJECT_ADDED, OBJECT_REMOVED);

struct Scene : Object {
    Scene() : Object("Scene") {
    }

    // compute good positions to place light and camera
    NODISCARD auto get_good_cam_start() const noexcept -> LookAtParams;
    NODISCARD auto get_good_light_pos() const noexcept -> glm::vec3;
    NODISCARD auto get_scene_bound() const noexcept -> BoundingBox;

    NODISCARD auto ray_cast(Ray const& ray, float t_min = 0.0f,
                            float t_max = 1e5f) noexcept -> ObserverPtr<SceneObject>;
    NODISCARD auto ray_cast_editable(Ray const& ray, float t_min = 0.0f,
                                     float t_max = 1e5f) noexcept -> ObserverPtr<SceneObject>;
    NODISCARD auto size() const noexcept {
        return m_size;
    }
    NODISCARD auto empty() const noexcept -> bool {
        return m_size == 0;
    }

    template <typename T, typename = std::enable_if_t<std::conjunction_v<
                              std::is_rvalue_reference<T&&>, std::is_base_of<SceneObject, T>,
                              Traits::is_reflectable<T>>>>
    auto add_object(T&& obj) noexcept -> ObserverPtr<T>;

    template <typename T,
              typename = std::enable_if_t<
                  std::conjunction_v<std::is_base_of<SceneObject, T>, Traits::is_reflectable<T>>>,
              typename... Args>
    auto emplace_object(Args&&... args) noexcept -> ObserverPtr<T>;

    template <typename T, typename = std::enable_if_t<std::conjunction_v<
                              std::is_base_of<SceneObject, T>, Traits::is_reflectable<T>>>>
    auto remove_object(T& obj) noexcept -> void;

    /**
     * @brief Gets all objects of the given type
     * @tparam T The type of objects to get
     * @return A list of all objects of the given type
     * @note There is no guarantee that the objects are in any particular order
     */
    template <typename T, typename = std::enable_if_t<std::conjunction_v<
                              std::is_base_of<SceneObject, T>, Traits::is_reflectable<T>>>>
    NODISCARD auto get_objects_of_type() const noexcept -> std::list<T> const&;

    /**
     * @brief Gets all objects that are editable, i.e. with edit flags ==
     * Visible | Selectable
     * @return A list of all editable objects
     * @note There is no guarantee that the objects are in any particular order
     */
    NODISCARD auto get_editables() const noexcept -> auto const& {
        return m_editables;
    }

    NODISCARD auto is_valid_obj(View<SceneObject> obj_view) const noexcept -> bool;

    NODISCARD auto next_obj_name() const noexcept -> std::string {
        static auto counter = 0;
        return "Object " + std::to_string(counter++);
    }

    NODISCARD auto next_light_name() const noexcept -> std::string {
        static auto counter = 0;
        return "Light " + std::to_string(counter++);
    }

    auto on_deserialize() noexcept -> void;
    auto try_add_editable(Ref<SceneObject> obj_view) noexcept -> void;
    auto try_remove_editable(View<SceneObject> obj_view) noexcept -> void;

    auto get_callback_list(SceneChangeType type) -> pts::Signal<void(Ref<SceneObject>)>&;

   private:
    BEGIN_REFLECT(Scene, Object);

    FIELD(std::list<RenderableObject>, m_renderable_objects, {}, MSerialize{},
          MNoInspect{});                                                // not editable
    FIELD(std::list<Light>, m_lights, {}, MSerialize{}, MNoInspect{});  // not editable
    FIELD(std::size_t, m_size, {}, MSerialize{}, MNoInspect{});         // not editable
    END_REFLECT();
    // enables dynamic retrieval of class info for polymorphic types
    DECL_DYNAMIC_INFO();

   private:
    // ----------- runtime-only data -----------
    // all objects that are editable
    std::list<Ref<SceneObject>> m_editables;

    // objects that are currently added to the scene
    std::unordered_set<ViewPtr<SceneObject>> m_alive_objs;

    EArray<SceneChangeType, pts::Signal<void(Ref<SceneObject>)>> m_scene_callbacks;
};

template <typename T, typename>
auto Scene::add_object(T&& obj) noexcept -> ObserverPtr<T> {
    return emplace_object<T>(std::move(obj));
}

template <typename T, typename, typename... Args>
auto Scene::emplace_object(Args&&... args) noexcept -> ObserverPtr<T> {
    auto ret = ObserverPtr<T>{};
    if constexpr (std::is_same_v<T, RenderableObject>) {
        ret = &m_renderable_objects.emplace_back(std::forward<Args>(args)...);
    } else if constexpr (std::is_same_v<T, Light>) {
        ret = &m_lights.emplace_back(std::forward<Args>(args)...);
    }

    m_alive_objs.emplace(ret);
    try_add_editable(*ret);

    m_scene_callbacks[SceneChangeType::OBJECT_ADDED](*ret);
    return ret;
}

template <typename T, typename>
auto Scene::remove_object(T& obj) noexcept -> void {
    if (is_valid_obj(obj)) {
        m_scene_callbacks[SceneChangeType::OBJECT_REMOVED](obj);

        if (auto const par = obj.get_parent()) {
            par->remove_child(obj);
        }

        try_remove_editable(obj);
        m_alive_objs.erase(&obj);

        // use runtime type cast here, because T can be polymorphic
        if (auto const prender_obj = dynamic_cast<RenderableObject const*>(&obj)) {
            m_renderable_objects.remove_if(
                [&](RenderableObject const& o) { return &o == prender_obj; });
        } else if (auto const plight = dynamic_cast<Light const*>(&obj)) {
            m_lights.remove_if([&](Light const& o) { return &o == plight; });
        }
    }
}

template <typename T, typename>
auto Scene::get_objects_of_type() const noexcept -> std::list<T> const& {
    if constexpr (std::is_same_v<T, RenderableObject>) {
        return m_renderable_objects;
    } else if constexpr (std::is_same_v<T, Light>) {
        return m_lights;
    } else {
        static_assert(false, "T must be either RenderableObject or Light for now");
    }
}
}  // namespace PTS
