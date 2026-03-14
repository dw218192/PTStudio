#include <core/rendering/adapters/registry.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

#include <unordered_map>

namespace pts::rendering {

namespace {

uint32_t extract_material(RenderWorld& world, const pxr::UsdPrim& prim,
                          std::unordered_map<std::string, uint32_t>& material_cache) {
    auto binding = pxr::UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
    if (!binding) return k_no_material;

    auto mat_path = binding.GetPath().GetString();

    auto it = material_cache.find(mat_path);
    if (it != material_cache.end()) return it->second;

    Material mat;

    auto surface = binding.ComputeSurfaceSource();
    if (surface) {
        pxr::TfToken shader_id;
        surface.GetShaderId(&shader_id);
        if (shader_id == pxr::TfToken("UsdPreviewSurface")) {
            if (auto input = surface.GetInput(pxr::TfToken("diffuseColor"))) {
                pxr::GfVec3f color;
                if (input.Get(&color)) mat.diffuse_color = {color[0], color[1], color[2]};
            }
            if (auto input = surface.GetInput(pxr::TfToken("metallic"))) {
                input.Get(&mat.metallic);
            }
            if (auto input = surface.GetInput(pxr::TfToken("roughness"))) {
                input.Get(&mat.roughness);
            }
            if (auto input = surface.GetInput(pxr::TfToken("opacity"))) {
                input.Get(&mat.opacity);
            }
        }
    }

    auto index = static_cast<uint32_t>(world.materials.size());
    world.materials.push_back(mat);
    material_cache[mat_path] = index;
    return index;
}

}  // namespace

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device) {
    std::unordered_map<std::string, uint32_t> material_cache;

    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        for (const auto* adapter : k_schema_adapters()) {
            if (!adapter->can_adapt(prim)) continue;

            auto result = adapter->adapt(prim);
            if (!result) continue;

            // Compute world transform (shared by all result types)
            pxr::GfMatrix4d xf = pxr::UsdGeomXformable(prim).ComputeLocalToWorldTransform(
                pxr::UsdTimeCode::Default());
            glm::mat4 transform;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) transform[j][i] = static_cast<float>(xf[i][j]);

            auto prim_path = prim.GetPath().GetString();

            std::visit(
                [&](auto& r) {
                    using T = std::decay_t<decltype(r)>;
                    if constexpr (std::is_same_v<T, MeshResult>) {
                        auto vertex_buf = device.create_buffer(
                            r.vertices.size() * sizeof(Vertex),
                            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex |
                                                         WGPUBufferUsage_CopyDst));
                        wgpuQueueWriteBuffer(device.queue(), vertex_buf.handle(), 0,
                                             r.vertices.data(),
                                             r.vertices.size() * sizeof(Vertex));

                        auto index_buf = device.create_buffer(
                            r.indices.size() * sizeof(uint32_t),
                            static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index |
                                                         WGPUBufferUsage_CopyDst));
                        wgpuQueueWriteBuffer(device.queue(), index_buf.handle(), 0,
                                             r.indices.data(),
                                             r.indices.size() * sizeof(uint32_t));

                        uint32_t mesh_index = static_cast<uint32_t>(world.meshes.size());

                        Mesh gpu_mesh;
                        gpu_mesh.vertex_buffer = std::move(vertex_buf);
                        gpu_mesh.index_buffer = std::move(index_buf);
                        gpu_mesh.index_count = static_cast<uint32_t>(r.indices.size());
                        world.meshes.push_back(std::move(gpu_mesh));

                        RenderObject obj;
                        obj.mesh_index = mesh_index;
                        obj.material_index = extract_material(world, prim, material_cache);
                        obj.transform = transform;
                        obj.prim_path = std::move(prim_path);
                        world.objects.push_back(std::move(obj));
                    } else if constexpr (std::is_same_v<T, LightResult>) {
                        Light light;
                        light.type = static_cast<Light::Type>(r.type);
                        light.color = r.color;
                        light.intensity = r.intensity;
                        light.direction = r.direction;
                        light.radius = r.radius;
                        light.width = r.width;
                        light.height = r.height;
                        light.transform = transform;
                        light.prim_path = std::move(prim_path);
                        world.lights.push_back(std::move(light));
                    }
                },
                *result);

            break;  // first matching adapter wins
        }
    }
}

}  // namespace pts::rendering
