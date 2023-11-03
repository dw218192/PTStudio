#pragma once

#include <tl/expected.hpp>
#include "transform.h"
#include "reflection.h"
#include "utils.h"

struct Light {
    Light() noexcept = default;
    Light(glm::vec3 color, float intensity) noexcept;

    NODISCARD auto get_color() const noexcept -> glm::vec3 const&;
    NODISCARD auto get_intensity() const noexcept -> float;
    NODISCARD auto get_transform() const noexcept -> Transform const&;

    void set_color(glm::vec3 color) noexcept;
    void set_intensity(float intensity) noexcept;
    void set_transform(Transform transform) noexcept;

private:
    BEGIN_REFLECT(Light);
        FIELD_MOD(glm::vec3, m_color,
                MSerialize{});
        FIELD_MOD(float, m_intensity,
                MSerialize{});
        FIELD_MOD(Transform, m_transform,
                MSerialize{});
    END_REFLECT();
};