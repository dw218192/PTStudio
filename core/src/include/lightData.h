#pragma once
#include <glm/glm.hpp>
#include <string_view>

namespace PTS {
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
}