#pragma once

#include "object.h"

#include <string>
#include <tl/expected.hpp>
#include <vector>

#include "camera.h"
#include "boundingBox.h"
#include "utils.h"
#include "reflection.h"

struct Scene {
    template<typename T>
    struct Iterator {
        static_assert(!std::is_same_v<typename std::iterator_traits<T>::iterator_category, void>,
            "T must be an iterator type");

        using value_type = typename T::pointer;
        using reference = value_type;
        using pointer = value_type;
        using iterator_category = std::bidirectional_iterator_tag;

        Iterator(T it) : m_it{ it } {}

        auto operator*() const -> reference { 
            return &*m_it;
        }
        auto operator++() -> Iterator& {
            ++m_it;
            return *this;
        }
        auto operator++(int) -> Iterator {
            auto ret = *this;
            ++m_it;
            return ret;
        }
        auto operator--() -> Iterator& {
            --m_it;
            return *this;
        }
        auto operator--(int) -> Iterator {
            auto ret = *this;
            --m_it;
            return ret;
        }
        auto operator==(const Iterator& other) const -> bool {
            return m_it == other.m_it;
        }
        auto operator!=(const Iterator& other) const -> bool {
            return m_it != other.m_it;
        }
    private:
        T m_it;
    };

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

    NODISCARD auto begin() const noexcept { return Iterator{ m_objects.begin() }; }
    NODISCARD auto end() const noexcept { return Iterator{ m_objects.end() }; }
    NODISCARD auto begin() noexcept { return Iterator{ m_objects.begin() }; }
    NODISCARD auto end() noexcept { return Iterator{ m_objects.end() }; }

    NODISCARD auto size() const noexcept { return m_objects.size(); }

    auto add_object(Object obj) noexcept -> ObserverPtr<Object>;
    void remove_object(ObserverPtr<Object> obj) noexcept;
    
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
    END_REFLECT();
};
