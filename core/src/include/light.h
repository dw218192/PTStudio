#pragma once

#include <tl/expected.hpp>
#include "transform.h"
#include "reflection.h"
#include "utils.h"

struct Scene;

struct Light {
    Light() noexcept = default;
    Light(Scene const& scene, glm::vec3 color, float intensity, Transform transform) noexcept;
    Light(glm::vec3 color, float intensity, Transform transform) noexcept;

    NODISCARD auto get_name() const noexcept -> std::string_view;
    NODISCARD auto get_color() const noexcept -> glm::vec3 const&;
    NODISCARD auto get_intensity() const noexcept -> float;
    NODISCARD auto get_transform() const noexcept -> Transform const&;

    void set_name(std::string_view name) noexcept;
    void set_color(glm::vec3 color) noexcept;
    void set_intensity(float intensity) noexcept;
    void set_transform(Transform transform) noexcept;

private:
    BEGIN_REFLECT(Light);
	    FIELD_MOD(std::string, m_name, "Light",
	        MSerialize{}, MNoInspect{}); // handled explicitly
        FIELD_MOD(glm::vec3, m_color, {},
            MSerialize{}, MColor{});
        FIELD_MOD(float, m_intensity, {},
            MSerialize{}, MRange{0.0f, 100.0f, 0.1f});
        FIELD_MOD(Transform, m_transform, {},
            MSerialize{}, MNoInspect{}); // handled explicitly
    END_REFLECT();
};
