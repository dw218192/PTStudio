#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <string_view>
#include <tcb/span.hpp>

#include "camera.h"
#include "lightData.h"
#include "material.h"
#include "vertex.h"

namespace PTS {
namespace Vk {
namespace VulkanRayTracingShaders {
struct Binding {
    int set, binding;
};

struct RayTracingBindings {
    static constexpr Binding ACCEL_STRUCT_BINDING{0, 0};
    static constexpr Binding MATERIALS_BINDING{0, 1};
    static constexpr Binding OUTPUT_IMAGE_BINDING{0, 2};
    static constexpr Binding LIGHTS_BINDING{0, 3};

    // variable sized SSBO must be in a separate descriptor set
    static constexpr Binding VERTEX_ATTRIBS_BINDING{1, 0};
    static constexpr Binding INDICES_BINDING{2, 0};

    static constexpr int k_num_bindings = 6;
    static constexpr int k_num_sets = 3;
};

struct CameraData {
    glm::mat4 inv_view_proj;  // 0
    glm::vec3 cam_pos;        // 64
    uint8_t _pad[4] = {0, 0, 0, 0};
    // total size: 80

    CameraData() = default;

    explicit CameraData(Camera const& camera)
        : inv_view_proj{camera.get_inv_view_proj()}, cam_pos{camera.get_eye()} {
    }

    auto operator==(CameraData const& other) const noexcept -> bool {
        return inv_view_proj == other.inv_view_proj && cam_pos == other.cam_pos;
    }

    auto operator!=(CameraData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view{
        "\
        struct CameraData {\n\
            mat4 inv_view_proj;\n\
            vec3 pos;\n\
        };\n"};
};

static_assert(sizeof(CameraData) == 80, "CameraData size mismatch");

struct PerFrameData {
    CameraData camera;              // 0  base alignment: 16
    uint32_t iteration;             // 80 base alignment: 4
    uint32_t num_samples;           // 84 base alignment: 4
    uint32_t max_bounces;           // 88 base alignment: 4
    uint32_t direct_lighting_only;  // 92 base alignment: 4
    // total size: 96

    PerFrameData(CameraData const& camera, int iteration, int num_samples, int max_bounces,
                 bool direct_lighting_only)
        : camera{camera},
          iteration{static_cast<uint32_t>(iteration)},
          num_samples{static_cast<uint32_t>(num_samples)},
          max_bounces{static_cast<uint32_t>(max_bounces)},
          direct_lighting_only{direct_lighting_only} {
    }

    static constexpr auto k_glsl_decl = std::string_view{
        "\
        struct PerFrameData {\n\
            CameraData camera;\n\
            int iteration;\n\
            int num_samples;\n\
            int max_bounces;\n\
			bool direct_lighting_only;\n\
        };\n"};
};

static_assert(offsetof(PerFrameData, camera) == 0, "PerFrameData::camera offset mismatch");
static_assert(offsetof(PerFrameData, iteration) == 80, "PerFrameData::iteration offset mismatch");
static_assert(offsetof(PerFrameData, num_samples) == 84,
              "PerFrameData::num_samples offset mismatch");
static_assert(offsetof(PerFrameData, max_bounces) == 88,
              "PerFrameData::max_bounces offset mismatch");
static_assert(offsetof(PerFrameData, direct_lighting_only) == 92,
              "PerFrameData::direct_lighting_only offset mismatch");
static_assert(sizeof(PerFrameData) == 96, "PerFrameData size mismatch");

struct VertexData {
    glm::vec3 position;

    VertexData() = default;
    explicit VertexData(Vertex const& vertex) : position{vertex.position} {
    }
};

struct MaterialData {
    glm::vec3 base_color;             // 0    base alignment: 16
    uint8_t _pad1[4] = {0, 0, 0, 0};  // 12
    glm::vec3 emissive_color;         // 16   base alignment: 16
    float emission_intensity;         // 28   base alignment: 4
    // total size: 32

    MaterialData() = default;

    explicit MaterialData(Material const& material)
        : base_color{material.albedo},
          emissive_color{material.emission},
          emission_intensity{material.emission_intensity} {
    }

    auto operator==(MaterialData const& other) const noexcept -> bool {
        return base_color == other.base_color && emissive_color == other.emissive_color &&
               std::abs(emission_intensity - other.emission_intensity) <= 0.001f;
    }

    auto operator!=(MaterialData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view{
        "\
        struct MaterialData {\n\
            vec3 base_color;\n\
            vec3 emissive_color;\n\
			float emission_intensity;\n\
        };\n"};
};

static_assert(sizeof(MaterialData) == 32, "MaterialData size mismatch");

struct LightBlock {
    static auto get_offset(size_t light_idx) {
        return 16 + light_idx * sizeof(LightData);
    }

    static auto get_mem(int num_lights, tcb::span<LightData> data) -> tcb::span<char> {
        auto const mem = reinterpret_cast<MemLayout*>(s_scratch_mem);
        mem->num_lights = num_lights;
        std::copy_n(data.data(), data.size(), mem->lights);
        return tcb::span{s_scratch_mem, get_offset(data.size())};
    }

   private:
    struct MemLayout {
        uint32_t num_lights;  // 0   base alignment: 4
        uint8_t _pad[12];     // 4
        LightData lights[1];  // 16  base alignment: 16
    };

    static inline char s_scratch_mem[16 + sizeof(LightData) * k_max_lights];
};

struct VertexAttribData {
    glm::vec3 normal;  // 0   base alignment: 16
    uint8_t _pad1[4] = {0, 0, 0, 0};
    glm::vec2 tex_coord;  // 16  base alignment: 8
    uint8_t _pad2[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    VertexAttribData() = default;

    explicit VertexAttribData(Vertex const& vertex) : normal{vertex.normal}, tex_coord{vertex.uv} {
    }

    auto operator==(VertexAttribData const& other) const noexcept -> bool {
        return normal == other.normal && tex_coord == other.tex_coord;
    }

    auto operator!=(VertexAttribData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view{
        "\
        struct VertexAttribData {\n\
            vec3 normal;\n\
            vec2 tex_coord;\n\
        };\n"};
};

static_assert(sizeof(VertexAttribData) == 32, "VertexAttribData size mismatch");

struct FaceIndexData {
    glm::uvec3 indices;
    uint8_t _pad[4] = {0, 0, 0, 0};

    FaceIndexData() = default;

    explicit FaceIndexData(unsigned i0, unsigned i1, unsigned i2) : indices{i0, i1, i2} {
    }
};

static_assert(sizeof(FaceIndexData) == 16, "FaceIndexData size mismatch");
}  // namespace VulkanRayTracingShaders
}  // namespace Vk
}  // namespace PTS
