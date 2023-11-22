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
        Spot,
        Mesh, // Object with emissive material
    };

    // used in glsl
    struct LightData {
        glm::vec3 color;        // 0    base alignment: 16
        float intensity;        // 12   base alignment: 4
        glm::vec3 position;     // 16   base alignment: 16
        int type;               // 28   base alignment: 4
        unsigned char _pad1[4] = { 0, 0, 0, 0 }; // 32
        // total size: 36

        static constexpr auto glsl_def = std::string_view{
            "struct LightData {\n"
            "   vec3 color;\n"
            "   float intensity;\n"
            "   vec3 position;\n"
            "   int type;\n"
            "};\n"
        };
    };

    static_assert(sizeof(LightData) == 36, "LightData size mismatch");

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

        auto get_data() const noexcept -> LightData {
            return {
                m_color,
                m_intensity,
                m_transform.get_position(),
                static_cast<int>(m_type),
            };
        }

    private:
        BEGIN_REFLECT(Light);
        FIELD_MOD(std::string, m_name, "Light",
            MSerialize{}, MNoInspect{}); // handled explicitly
        FIELD_MOD(LightType, m_type, LightType::Point,
            MSerialize{},
            MEnum {
                [] (int idx) -> char const* {
                    switch (idx) {
                    case 0: return "Directional";
                    case 1: return "Point";
                    case 2: return "Spot";
                    case 3: return "Mesh";
                    default: return "Unknown";
                    }
                },
                4
            }
        );
        FIELD_MOD(glm::vec3, m_color, {},
            MSerialize{}, MColor{});
        FIELD_MOD(float, m_intensity, {},
            MSerialize{}, MRange{ 0.0f, 100.0f });
        FIELD_MOD(Transform, m_transform, {},
            MSerialize{}, MNoInspect{}); // handled explicitly
        END_REFLECT();
    };
}