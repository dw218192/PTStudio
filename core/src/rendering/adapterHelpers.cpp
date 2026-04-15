#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

#include <cmath>

namespace pts::rendering {

namespace {

uint32_t channel_from_output_name(const pxr::TfToken& name) {
    static const pxr::TfToken k_r("r"), k_R("R");
    static const pxr::TfToken k_g("g"), k_G("G");
    static const pxr::TfToken k_b("b"), k_B("B");
    static const pxr::TfToken k_a("a"), k_A("A");
    if (name == k_r || name == k_R) return 0;
    if (name == k_g || name == k_G) return 1;
    if (name == k_b || name == k_B) return 2;
    if (name == k_a || name == k_A) return 3;
    return 0;
}

/// Try to resolve a texture connection on a UsdShadeInput.
/// Returns texture layer index (UINT32_MAX if no texture connected or load failed).
/// Sets out_source_name to the output name of the UsdUVTexture (for channel selection).
uint32_t try_resolve_texture(pxr::UsdShadeInput input, SyncScope& scope,
                             pxr::TfToken& out_source_name) {
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken source_name;
    pxr::UsdShadeAttributeType source_type;
    if (!input.GetConnectedSource(&source, &source_name, &source_type)) {
        return UINT32_MAX;
    }

    auto shader = pxr::UsdShadeShader(source.GetPrim());
    pxr::TfToken shader_id;
    shader.GetShaderId(&shader_id);
    if (shader_id != pxr::TfToken("UsdUVTexture")) {
        return UINT32_MAX;
    }

    pxr::SdfAssetPath file_path;
    auto file_input = shader.GetInput(pxr::TfToken("file"));
    if (!file_input || !file_input.Get(&file_path)) {
        return UINT32_MAX;
    }

    std::string resolved = file_path.GetResolvedPath();
    if (resolved.empty()) {
        resolved = pxr::ArGetResolver().Resolve(file_path.GetAssetPath()).GetPathString();
    }
    if (resolved.empty()) {
        return UINT32_MAX;
    }
    out_source_name = source_name;
    return scope.load_texture(resolved);
}

}  // namespace

glm::mat4 compute_world_transform(pxr::UsdPrim prim) {
    pxr::GfMatrix4d xf =
        pxr::UsdGeomXformable(prim).ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default());
    glm::mat4 transform;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) transform[i][j] = static_cast<float>(xf[i][j]);
    return transform;
}

Material read_preview_surface(pxr::UsdShadeShader surface, SyncScope& scope) {
    Material mat;

    // --- diffuseColor ---
    if (auto input = surface.GetInput(pxr::TfToken("diffuseColor"))) {
        pxr::GfVec3f color;
        if (input.Get(&color)) mat.diffuse_color = {color[0], color[1], color[2]};
        pxr::TfToken source_name;
        auto tex = try_resolve_texture(input, scope, source_name);
        if (tex != UINT32_MAX) mat.diffuse_tex = tex;
    }

    // --- metallic ---
    uint32_t metallic_ch = 0;
    if (auto input = surface.GetInput(pxr::TfToken("metallic"))) {
        input.Get(&mat.metallic);
        pxr::TfToken source_name;
        auto tex = try_resolve_texture(input, scope, source_name);
        if (tex != UINT32_MAX) {
            mat.metallic_tex = tex;
            metallic_ch = channel_from_output_name(source_name);
        }
    }

    // --- roughness ---
    uint32_t roughness_ch = 0;
    if (auto input = surface.GetInput(pxr::TfToken("roughness"))) {
        input.Get(&mat.roughness);
        pxr::TfToken source_name;
        auto tex = try_resolve_texture(input, scope, source_name);
        if (tex != UINT32_MAX) {
            mat.roughness_tex = tex;
            roughness_ch = channel_from_output_name(source_name);
        }
    }

    // --- opacity ---
    uint32_t opacity_ch = 0;
    if (auto input = surface.GetInput(pxr::TfToken("opacity"))) {
        input.Get(&mat.opacity);
        pxr::TfToken source_name;
        auto tex = try_resolve_texture(input, scope, source_name);
        if (tex != UINT32_MAX) {
            mat.opacity_tex = tex;
            opacity_ch = channel_from_output_name(source_name);
        }
    }

    // --- ior ---
    if (auto input = surface.GetInput(pxr::TfToken("ior"))) {
        input.Get(&mat.ior);
    }

    // --- opacityThreshold ---
    if (auto input = surface.GetInput(pxr::TfToken("opacityThreshold"))) {
        input.Get(&mat.opacity_threshold);
    }

    // --- normal ---
    if (auto input = surface.GetInput(pxr::TfToken("normal"))) {
        pxr::TfToken source_name;
        auto tex = try_resolve_texture(input, scope, source_name);
        if (tex != UINT32_MAX) mat.normal_tex = tex;
    }

    // --- emissiveColor ---
    if (auto input = surface.GetInput(pxr::TfToken("emissiveColor"))) {
        pxr::GfVec3f color;
        if (input.Get(&color)) mat.emissive_color = {color[0], color[1], color[2]};
        pxr::TfToken source_name;
        auto tex = try_resolve_texture(input, scope, source_name);
        if (tex != UINT32_MAX) mat.emissive_tex = tex;
    }

    mat.tex_channels = metallic_ch | (roughness_ch << 2) | (opacity_ch << 4);
    return mat;
}

uint32_t resolve_material(pxr::UsdPrim prim, SyncScope& scope) {
    auto binding = pxr::UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
    if (!binding) {
        return k_no_material;
    }

    auto mat_path = binding.GetPath().GetString();

    auto& cache = scope.material_cache();
    auto it = cache.find(mat_path);
    if (it != cache.end()) {
        return it->second;
    }

    Material mat;

    auto surface = binding.ComputeSurfaceSource();
    if (surface) {
        pxr::TfToken shader_id;
        surface.GetShaderId(&shader_id);
        if (shader_id == pxr::TfToken("UsdPreviewSurface")) {
            mat = read_preview_surface(surface, scope);
        }
    }

    auto& materials = scope.materials();
    auto index = static_cast<uint32_t>(materials.size());
    materials.push_back(mat);
    cache[mat_path] = index;
    return index;
}

void store_mesh(SyncScope& scope, const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices, uint32_t mesh_slot) {
    scope.mutate_mesh(mesh_slot, [&](MeshData& w) {
        w.cpu_vertices.assign(vertices.begin(), vertices.end());
        w.cpu_indices.assign(indices.begin(), indices.end());
        w.index_count = static_cast<uint32_t>(indices.size());
    });
}

void sync_object(pxr::UsdPrim geom_prim, const pxr::SdfPath& obj_path, uint32_t material_index,
                 SyncScope& scope, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    auto& world = scope.world();
    auto transform = compute_world_transform(geom_prim);
    auto vis = pxr::UsdGeomImageable(geom_prim).ComputeVisibility();
    bool visible = (vis != pxr::UsdGeomTokens->invisible);

    int existing = world.find_object_by_prim(obj_path);
    if (existing >= 0) {
        auto mesh_index = scope.object(static_cast<uint32_t>(existing)).mesh_index;
        scope.mutate_object(static_cast<uint32_t>(existing), [&](ObjectData& w) {
            w.transform = transform;
            w.material_index = material_index;
            w.visible = visible;
        });
        store_mesh(scope, vertices, indices, mesh_index);
    } else {
        auto mesh_slot = scope.alloc_mesh(obj_path);
        auto obj_slot = scope.alloc_object(obj_path);
        store_mesh(scope, vertices, indices, mesh_slot);
        scope.mutate_object(obj_slot, [&](ObjectData& w) {
            w.mesh_index = mesh_slot;
            w.transform = transform;
            w.material_index = material_index;
            w.visible = visible;
        });
    }
}

void sync_object(pxr::UsdPrim prim, SyncScope& scope, std::vector<Vertex>& vertices,
                 std::vector<uint32_t>& indices) {
    auto material_index = resolve_material(prim, scope);

    if (material_index == k_no_material) {
        auto sdf_path = prim.GetPath();
        auto colors = read_display_color(prim);
        if (!colors.empty()) {
            auto cache_key = "$displayColor:" + sdf_path.GetString();
            auto& cache = scope.material_cache();
            auto it = cache.find(cache_key);
            if (it != cache.end()) {
                material_index = it->second;
                auto& mat = scope.materials()[material_index];
                mat.diffuse_color = {colors[0][0], colors[0][1], colors[0][2]};
            } else {
                Material mat;
                mat.diffuse_color = {colors[0][0], colors[0][1], colors[0][2]};
                auto& materials = scope.materials();
                material_index = static_cast<uint32_t>(materials.size());
                materials.push_back(mat);
                cache[cache_key] = material_index;
            }
        } else {
            material_index = k_default_material;
        }
    }

    sync_object(prim, prim.GetPath(), material_index, scope, vertices, indices);
}

// --- Proxy mesh geometry generators ---

static constexpr float k_pi = 3.14159265358979323846f;

void generate_rect_mesh(float width, float height, std::vector<Vertex>& out_vertices,
                        std::vector<uint32_t>& out_indices) {
    float hw = width * 0.5f;
    float hh = height * 0.5f;
    // Quad centered at origin, facing -Z (USD lights emit along -Z)
    out_vertices = {
        {{-hw, -hh, 0.0f}, {0, 0, -1}, {1, 1, 1}, {0, 0}},
        {{hw, -hh, 0.0f}, {0, 0, -1}, {1, 1, 1}, {1, 0}},
        {{hw, hh, 0.0f}, {0, 0, -1}, {1, 1, 1}, {1, 1}},
        {{-hw, hh, 0.0f}, {0, 0, -1}, {1, 1, 1}, {0, 1}},
    };
    out_indices = {0, 2, 1, 0, 3, 2};
}

void generate_disk_mesh(float radius, std::vector<Vertex>& out_vertices,
                        std::vector<uint32_t>& out_indices) {
    static constexpr uint32_t k_segments = 48;
    // Disk centered at origin, facing -Z (USD lights emit along -Z)
    out_vertices.push_back({{0, 0, 0}, {0, 0, -1}, {1, 1, 1}, {0.5f, 0.5f}});

    for (uint32_t i = 0; i <= k_segments; ++i) {
        float theta = 2.0f * k_pi * static_cast<float>(i) / static_cast<float>(k_segments);
        float cx = std::cos(theta);
        float cy = std::sin(theta);
        out_vertices.push_back({
            {radius * cx, radius * cy, 0.0f},
            {0, 0, -1},
            {1, 1, 1},
            {cx * 0.5f + 0.5f, cy * 0.5f + 0.5f},
        });
    }

    // Triangle fan (reversed winding for -Z facing)
    for (uint32_t i = 0; i < k_segments; ++i) {
        out_indices.push_back(0);
        out_indices.push_back(i + 2);
        out_indices.push_back(i + 1);
    }
}

void generate_sphere_mesh(float radius, std::vector<Vertex>& out_vertices,
                          std::vector<uint32_t>& out_indices) {
    static constexpr uint32_t k_lon = 16;
    static constexpr uint32_t k_lat = 8;

    for (uint32_t lat = 0; lat <= k_lat; ++lat) {
        float phi = k_pi * static_cast<float>(lat) / static_cast<float>(k_lat);
        float sp = std::sin(phi);
        float cp = std::cos(phi);

        for (uint32_t lon = 0; lon <= k_lon; ++lon) {
            float theta = 2.0f * k_pi * static_cast<float>(lon) / static_cast<float>(k_lon);
            float st = std::sin(theta);
            float ct = std::cos(theta);

            float nx = sp * ct;
            float ny = cp;
            float nz = sp * st;

            float u = static_cast<float>(lon) / static_cast<float>(k_lon);
            float v = static_cast<float>(lat) / static_cast<float>(k_lat);

            out_vertices.push_back({
                {radius * nx, radius * ny, radius * nz},
                {nx, ny, nz},
                {1, 1, 1},
                {u, v},
            });
        }
    }

    for (uint32_t lat = 0; lat < k_lat; ++lat) {
        for (uint32_t lon = 0; lon < k_lon; ++lon) {
            uint32_t a = lat * (k_lon + 1) + lon;
            uint32_t b = a + k_lon + 1;
            out_indices.push_back(a);
            out_indices.push_back(b);
            out_indices.push_back(a + 1);
            out_indices.push_back(a + 1);
            out_indices.push_back(b);
            out_indices.push_back(b + 1);
        }
    }
}

static uint32_t resolve_emissive_material(SyncScope& scope, const std::string& prim_path,
                                          const glm::vec3& color, float intensity) {
    auto cache_key = "$lightEmissive:" + prim_path;
    auto& cache = scope.material_cache();
    auto it = cache.find(cache_key);
    if (it != cache.end()) {
        auto& mat = scope.materials()[it->second];
        mat.emissive_color = color * intensity;
        return it->second;
    }

    Material mat;
    mat.diffuse_color = {0, 0, 0};
    mat.emissive_color = color * intensity;
    auto& materials = scope.materials();
    auto index = static_cast<uint32_t>(materials.size());
    materials.push_back(mat);
    cache[cache_key] = index;
    return index;
}

static bool light_type_has_proxy(LightData::Type type) {
    return type == LightData::Type::Rect || type == LightData::Type::Disk ||
           type == LightData::Type::Sphere;
}

static void generate_proxy_mesh(const LightData& light, std::vector<Vertex>& vertices,
                                std::vector<uint32_t>& indices) {
    switch (light.type) {
        case LightData::Type::Rect:
            generate_rect_mesh(light.width, light.height, vertices, indices);
            break;
        case LightData::Type::Disk:
            generate_disk_mesh(light.radius, vertices, indices);
            break;
        case LightData::Type::Sphere:
            generate_sphere_mesh(light.radius, vertices, indices);
            break;
        default:
            UNREACHABLE();
    }
}

void sync_light(pxr::UsdPrim prim, SyncScope& scope, const LightData& light) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();

    int existing = world.find_light_by_prim(sdf_path);
    if (existing >= 0) {
        auto prev_mesh = scope.light(static_cast<uint32_t>(existing)).mesh_index;
        auto prev_mat = scope.light(static_cast<uint32_t>(existing)).material_index;

        // Pre-compute proxy mesh resources outside the mutation
        uint32_t new_mat = prev_mat;
        uint32_t new_mesh = prev_mesh;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        if (light_type_has_proxy(light.type)) {
            new_mat = resolve_emissive_material(scope, sdf_path.GetString(), light.color,
                                                light.intensity);
            generate_proxy_mesh(light, vertices, indices);

            if (prev_mesh == UINT32_MAX) {
                new_mesh = scope.alloc_mesh(sdf_path);
            }
        }

        scope.mutate_light(static_cast<uint32_t>(existing), [&](LightData& w) {
            w = light;
            w.mesh_index = new_mesh;
            w.material_index = new_mat;
        });

        if (light_type_has_proxy(light.type)) {
            store_mesh(scope, vertices, indices, new_mesh);
        }
    } else {
        auto slot = scope.alloc_light(sdf_path);

        // Pre-compute proxy mesh resources outside the mutation
        uint32_t mat_idx = light.material_index;
        uint32_t mesh_idx = light.mesh_index;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        if (light_type_has_proxy(light.type)) {
            mat_idx = resolve_emissive_material(scope, sdf_path.GetString(), light.color,
                                                light.intensity);
            generate_proxy_mesh(light, vertices, indices);
            mesh_idx = scope.alloc_mesh(sdf_path);
        }

        scope.mutate_light(slot, [&](LightData& w) {
            w = light;
            w.material_index = mat_idx;
            w.mesh_index = mesh_idx;
        });

        if (light_type_has_proxy(light.type)) {
            store_mesh(scope, vertices, indices, mesh_idx);
        }
    }
}

void sync_camera(pxr::UsdPrim prim, SyncScope& scope, const CameraData& camera) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();

    int existing = world.find_camera_by_prim(sdf_path);
    if (existing >= 0) {
        scope.mutate_camera(static_cast<uint32_t>(existing), [&](CameraData& w) { w = camera; });
    } else {
        auto slot = scope.alloc_camera(sdf_path);
        scope.mutate_camera(slot, [&](CameraData& w) { w = camera; });
    }
}

}  // namespace pts::rendering
