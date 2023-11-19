#pragma once

#include <tl/expected.hpp>
#include "transform.h"
#include "reflection.h"
#include "utils.h"


namespace PTS {
    struct Scene;

    enum class LightType {
        Directional,
        Point,
        Spot
    };

    struct Light {
        Light() noexcept = default;
        Light(Scene const& scene, glm::vec3 color, float intensity, Transform transform) noexcept;
        Light(glm::vec3 color, float intensity, Transform transform) noexcept;

        NODISCARD auto get_name() const noexcept -> std::string_view { return m_name; }
        NODISCARD auto get_color() const noexcept -> auto const& { return m_color; }
        NODISCARD auto get_intensity() const noexcept { return m_intensity; }
        NODISCARD auto get_transform() const noexcept -> auto const& { return m_transform; }

        void set_name(std::string_view name) noexcept { m_name = name; }
        void set_color(glm::vec3 color) noexcept { m_color = color; }
        void set_intensity(float intensity) noexcept { m_intensity = intensity; }
        void set_transform(Transform transform) noexcept { m_transform = std::move(transform); }

    private:
        BEGIN_REFLECT(Light);
        FIELD_MOD(std::string, m_name, "Light",
            MSerialize{}, MNoInspect{}); // handled explicitly
        FIELD_MOD(LightType, m_type, LightType::Point,
            MSerialize{}, MNoInspect{}); // handled explicitly
        FIELD_MOD(glm::vec3, m_color, {},
            MSerialize{}, MColor{});
        FIELD_MOD(float, m_intensity, {},
            MSerialize{}, MRange{ 0.0f, 100.0f, 0.1f });
        FIELD_MOD(Transform, m_transform, {},
            MSerialize{}, MNoInspect{}); // handled explicitly
        END_REFLECT();
    };
}