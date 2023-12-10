#pragma once

#include <tl/expected.hpp>
#include "reflection.h"
#include "utils.h"
#include "sceneObject.h"
#include "lightData.h"

namespace PTS {
    struct Light : SceneObject {
        Light(ObjectConstructorUsage usage = ObjectConstructorUsage::SERIALIZE) noexcept;
        Light(Scene const& scene, Transform transform, glm::vec3 color, float intensity) noexcept;

        NODISCARD auto get_color() const noexcept -> auto const& { return m_color; }
        NODISCARD auto get_intensity() const noexcept { return m_intensity; }
        void set_color(glm::vec3 color) noexcept { m_color = color; }
        void set_intensity(float intensity) noexcept { m_intensity = intensity; }

        auto get_data() const noexcept -> LightData {
            return {
                m_color,
                m_intensity,
                get_transform().get_position(),
                static_cast<int>(m_type),
            };
        }

    private:
        BEGIN_REFLECT(Light, SceneObject);
        FIELD(LightType, m_type, LightType::Point,
            MSerialize{},
            MEnum {
                4,
                [] (int idx) -> char const* {
                    switch (idx) {
                    case 0: return "Directional";
                    case 1: return "Point";
                    case 2: return "Spot";
                    case 3: return "Mesh";
                    default: return "Unknown";
                    }
                }
            }
        );
        FIELD(glm::vec3, m_color, {},
            MSerialize{}, MColor{});
        FIELD(float, m_intensity, {},
            MSerialize{}, MRange{ 0.0f, 100.0f });
        END_REFLECT();

        // enables dynamic retrieval of class info for polymorphic types
        DECL_DYNAMIC_INFO();
    };
}