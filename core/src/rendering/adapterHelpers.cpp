#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

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
    return 0;  // "rgb" or default → red channel fallback for scalar
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
    auto w = scope.write_mesh(mesh_slot);
    w->cpu_vertices.assign(vertices.begin(), vertices.end());
    w->cpu_indices.assign(indices.begin(), indices.end());
    w->index_count = static_cast<uint32_t>(indices.size());
}

void sync_object(pxr::UsdPrim prim, SyncScope& scope, std::vector<Vertex>& vertices,
                 std::vector<uint32_t>& indices) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();
    auto transform = compute_world_transform(prim);
    auto material_index = resolve_material(prim, scope);

    if (material_index == k_no_material) {
        auto colors = read_display_color(prim);
        if (!colors.empty()) {
            // Use material cache with a synthetic key to avoid unbounded
            // growth when the same prim is re-synced.
            auto cache_key = "$displayColor:" + sdf_path.GetString();
            auto& cache = scope.material_cache();
            auto it = cache.find(cache_key);
            if (it != cache.end()) {
                material_index = it->second;
                auto& mat = scope.materials()[material_index - 1];
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

    int existing = world.find_object_by_prim(sdf_path);
    if (existing >= 0) {
        auto w = scope.write_object(static_cast<uint32_t>(existing));
        auto mesh_index = w->mesh_index;
        w->transform = transform;
        w->material_index = material_index;
        store_mesh(scope, vertices, indices, mesh_index);
    } else {
        auto mesh_slot = scope.alloc_mesh_slot();
        auto obj_slot = scope.alloc_object_slot();
        store_mesh(scope, vertices, indices, mesh_slot);
        {
            auto w = scope.write_object(obj_slot);
            w->mesh_index = mesh_slot;
            w->transform = transform;
            w->material_index = material_index;
        }
        scope.set_prim_path(obj_slot, PrimSlot::Kind::Object, sdf_path);
    }
}

void sync_light(pxr::UsdPrim prim, SyncScope& scope, const LightData& light) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();

    int existing = world.find_light_by_prim(sdf_path);
    if (existing >= 0) {
        auto w = scope.write_light(static_cast<uint32_t>(existing));
        *w = light;
    } else {
        auto slot = scope.alloc_light_slot();
        {
            auto w = scope.write_light(slot);
            *w = light;
        }
        scope.set_prim_path(slot, PrimSlot::Kind::Light, sdf_path);
    }
}

void sync_camera(pxr::UsdPrim prim, SyncScope& scope, const CameraData& camera) {
    auto& world = scope.world();
    auto sdf_path = prim.GetPath();

    int existing = world.find_camera_by_prim(sdf_path);
    if (existing >= 0) {
        auto w = scope.write_camera(static_cast<uint32_t>(existing));
        *w = camera;
    } else {
        auto slot = scope.alloc_camera_slot();
        {
            auto w = scope.write_camera(slot);
            *w = camera;
        }
        scope.set_prim_path(slot, PrimSlot::Kind::Camera, sdf_path);
    }
}

}  // namespace pts::rendering
