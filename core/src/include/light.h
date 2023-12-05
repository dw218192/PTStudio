#pragma once

#include <tl/expected.hpp>
#include "reflection.h"
#include "utils.h"
#include "sceneObject.h"


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
        FIELD(std::string, m_name, "Light",
            MSerialize{}, MNoInspect{}); // handled explicitly
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