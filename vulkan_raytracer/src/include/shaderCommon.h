#pragma once
#include <glm/glm.hpp>
#include <string_view>
#include "camera.h"
#include "vertex.h"
#include "material.h"
#include "stringManip.h"


namespace PTS {

struct CameraData {
    glm::mat4 inv_view_proj; // 0
    glm::vec3 cam_pos;       // 64
    unsigned char _pad[4] = { 0, 0, 0, 0 };
    // total size: 80

    CameraData() = default;
    explicit CameraData(Camera const& camera) : 
        inv_view_proj{camera.get_inv_view_proj()}, cam_pos{camera.get_eye()} {}

    auto operator==(CameraData const& other) const noexcept -> bool {
        return inv_view_proj == other.inv_view_proj
            && cam_pos == other.cam_pos;
    }
    auto operator!=(CameraData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct CameraData {\n\
            mat4 inv_view_proj;\n\
            vec3 pos;\n\
        };\n" };
};
static_assert(sizeof(CameraData) == 80, "CameraData size mismatch");

struct PerFrameData {
    CameraData camera; // 0
    int iteration;     // 80
    int num_samples;   // 84
    int max_bounces;   // 88
    // total size: 96

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct PerFrameData {\n\
            CameraData camera;\n\
            int iteration;\n\
            int num_samples;\n\
            int max_bounces;\n\
        };\n" };
};

struct VertexData {
    glm::vec3 position;

    VertexData() = default;
    explicit VertexData(Vertex const& vertex) : position{vertex.position} {}
};
struct MaterialData {
    glm::vec3 base_color;                          // 0    base alignment: 16
    unsigned char _pad1[4] = { 0, 0, 0, 0 };       // 12
    glm::vec3 emissive_color;                      // 16   base alignment: 16
    unsigned char _pad2[4] = { 0, 0, 0, 0 };       // 28
    // total size: 32

    MaterialData() = default;
    explicit MaterialData(Material const& material) : 
        base_color{material.albedo}, emissive_color{material.emission} {}

    auto operator==(MaterialData const& other) const noexcept -> bool {
        return base_color == other.base_color
            && emissive_color == other.emissive_color;
    }
    auto operator!=(MaterialData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct MaterialData {\n\
            vec3 base_color;\n\
            vec3 emissive_color;\n\
        };\n" };
};
static_assert(sizeof(MaterialData) == 32, "MaterialData size mismatch");

struct VertexAttribData {
    glm::vec3 normal;     // 0   base alignment: 16
    unsigned char _pad1[4] = { 0, 0, 0, 0 };
    glm::vec2 tex_coord;  // 16  base alignment: 8
    unsigned char _pad2[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    VertexAttribData() = default;
    explicit VertexAttribData(Vertex const& vertex) : 
        normal{vertex.normal}, tex_coord{vertex.uv} {}

    auto operator==(VertexAttribData const& other) const noexcept -> bool {
        return normal == other.normal
            && tex_coord == other.tex_coord;
    }
    auto operator!=(VertexAttribData const& other) const noexcept -> bool {
        return !(*this == other);
    }

    static constexpr auto k_glsl_decl = std::string_view { "\
        struct VertexAttribData {\n\
            vec3 normal;\n\
            vec2 tex_coord;\n\
        };\n" };
};
static_assert(sizeof(VertexAttribData) == 32, "VertexAttribData size mismatch");

struct FaceIndexData {
    glm::uvec3 indices;
    unsigned char _pad[4] = { 0, 0, 0, 0 };

    FaceIndexData() = default;
    explicit FaceIndexData(unsigned i0, unsigned i1, unsigned i2) : 
        indices{i0, i1, i2} {} 
};
static_assert(sizeof(FaceIndexData) == 16, "FaceIndexData size mismatch");

struct Binding {
    int set, binding;
};
struct RayTracingBindings {
    static constexpr Binding ACCEL_STRUCT_BINDING { 0, 0 };
    static constexpr Binding MATERIALS_BINDING { 0, 1 };
    static constexpr Binding OUTPUT_IMAGE_BINDING { 0, 2 };

    static constexpr Binding VERTEX_ATTRIBS_BINDING { 1, 0 };
    static constexpr Binding INDICES_BINDING { 2, 0 };

    static constexpr int k_num_bindings = 5;
    static constexpr int k_num_sets = 3;
};






namespace _private {
// silence MSVC warning C4455 which is a bug
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4455)
using std::literals::string_view_literals::operator""sv;
#pragma warning(pop)
#endif

// meta function to generate glsl declaration
template <int set, int binding, std::string_view const& decl, std::string_view const&... mods>
struct gen_glsl_decl {
    static auto constexpr _0 = "layout (set ="sv;
    static auto constexpr _1 = ", binding ="sv;
    static auto constexpr _2 = ") "sv;
    static auto constexpr _3 = ", "sv;
    static auto constexpr impl() {
        if constexpr (sizeof...(mods) == 0) {
            return PTS::join_v<_0, PTS::to_str_v<set>, _1, PTS::to_str_v<binding>, _2, decl>;
        } else {
            return PTS::join_v<_0, PTS::to_str_v<set>, _1, PTS::to_str_v<binding>, _3, PTS::join_v<mods...>, _2, decl>;
        }
    }
    static auto constexpr value = impl();
};

auto constexpr _k_vertex_attribs_decl = R"(
readonly buffer VertexAttribsBlock {
    VertexAttribData data[];
} vertexAttribs[];
)"sv;
auto constexpr k_vertex_attribs_decl = gen_glsl_decl<
    RayTracingBindings::VERTEX_ATTRIBS_BINDING.set,
    RayTracingBindings::VERTEX_ATTRIBS_BINDING.binding,
    _k_vertex_attribs_decl
>::value;

// indices declaration
auto constexpr _k_indices_decl = R"(
readonly buffer FaceIndicesBlock {
    uvec3 data[];
} faceIndices[];
)"sv;
auto constexpr k_indices_decl = gen_glsl_decl<
    RayTracingBindings::INDICES_BINDING.set,
    RayTracingBindings::INDICES_BINDING.binding,
    _k_indices_decl
>::value;

// material uniform declaration
auto constexpr _k_material_uniform_decl_0 = R"(
uniform MaterialBlock {
    MaterialData materials[)"sv;
auto constexpr _k_material_uniform_decl_1 = R"(];
};
)"sv;
auto constexpr _k_material_uniform_decl = PTS::join_v<
    _k_material_uniform_decl_0,
    PTS::to_str_v<k_max_instances>,
    _k_material_uniform_decl_1
>;
auto constexpr k_material_uniform_decl = gen_glsl_decl<
    RayTracingBindings::MATERIALS_BINDING.set,
    RayTracingBindings::MATERIALS_BINDING.binding,
    _k_material_uniform_decl
>::value;

// camera uniform declaration
auto constexpr k_per_frame_data_decl = R"(
layout (push_constant) uniform PerFrameDataBlock {
    PerFrameData perFrameData;
};
)"sv;

// output image declaration
auto constexpr _k_output_image_decl = R"(
uniform image2D outputImage;
)"sv;
auto constexpr _k_output_image_mod = "rgba8"sv;
auto constexpr k_output_image_decl = gen_glsl_decl<
    RayTracingBindings::OUTPUT_IMAGE_BINDING.set,
    RayTracingBindings::OUTPUT_IMAGE_BINDING.binding,
    _k_output_image_decl,
    _k_output_image_mod
>::value;

// acceleration structure declaration
auto constexpr _k_accel_struct_decl = R"(
uniform accelerationStructureEXT topLevelAS;
)"sv;

auto constexpr k_accel_struct_decl = gen_glsl_decl<
    RayTracingBindings::ACCEL_STRUCT_BINDING.set,
    RayTracingBindings::ACCEL_STRUCT_BINDING.binding,
    _k_accel_struct_decl
>::value;

} // namespace _private
} // namespace PTS