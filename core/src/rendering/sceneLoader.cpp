#include <core/profiling.h>
#include <core/rendering/adapters/registry.h>
#include <core/rendering/renderWorld.h>
#include <core/rendering/sceneLoader.h>
#include <pxr/usd/usd/primRange.h>

namespace pts::rendering {

namespace {

void upload_mesh(RenderWorld& world, const webgpu::Device& device, const MeshResult& mesh,
                 RenderObject& obj) {
    auto vertex_buf = device.create_buffer(
        mesh.vertices.size() * sizeof(Vertex),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst));
    wgpuQueueWriteBuffer(device.queue(), vertex_buf.handle(), 0, mesh.vertices.data(),
                         mesh.vertices.size() * sizeof(Vertex));

    auto index_buf = device.create_buffer(
        mesh.indices.size() * sizeof(uint32_t),
        static_cast<WGPUBufferUsage>(WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst));
    wgpuQueueWriteBuffer(device.queue(), index_buf.handle(), 0, mesh.indices.data(),
                         mesh.indices.size() * sizeof(uint32_t));

    obj.mesh_index = static_cast<uint32_t>(world.meshes.size());

    Mesh gpu_mesh;
    gpu_mesh.vertex_buffer = std::move(vertex_buf);
    gpu_mesh.index_buffer = std::move(index_buf);
    gpu_mesh.index_count = static_cast<uint32_t>(mesh.indices.size());
    gpu_mesh.cpu_indices.assign(mesh.indices.begin(), mesh.indices.end());
    world.meshes.push_back(std::move(gpu_mesh));
}

}  // namespace

void populate_from_stage(RenderWorld& world, const pxr::UsdStageRefPtr& stage,
                         const webgpu::Device& device) {
    PTS_ZONE_SCOPED;
    ++world.mesh_version;
    for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
        for (const auto* adapter : k_schema_adapters()) {
            if (!adapter->can_adapt(prim)) continue;

            auto prim_path = prim.GetPath().GetString();

            // Build a lightweight RenderObject and run property adapters.
            // Property adapters can Skip to discard this prim entirely.
            RenderObject obj;
            obj.prim_path = prim_path;

            bool skipped = false;
            for (const auto* prop : k_property_adapters()) {
                if (prop->apply(prim, obj, world) == AdapterAction::Skip) {
                    skipped = true;
                    break;
                }
            }
            if (skipped) break;

            // Typed adaptation (tessellation / light extraction)
            auto result = adapter->adapt(prim);
            if (!result) break;

            std::visit(
                [&](auto& r) {
                    using T = std::decay_t<decltype(r)>;
                    if constexpr (std::is_same_v<T, MeshResult>) {
                        upload_mesh(world, device, r, obj);
                        auto obj_path = obj.prim_path;
                        world.objects.push_back(std::move(obj));
                        world.prim_to_object[std::move(obj_path)] =
                            static_cast<uint32_t>(world.objects.size() - 1);
                    } else if constexpr (std::is_same_v<T, LightResult>) {
                        Light light;
                        light.type = static_cast<Light::Type>(r.type);
                        light.color = r.color;
                        light.intensity = r.intensity;
                        light.direction = r.direction;
                        light.radius = r.radius;
                        light.width = r.width;
                        light.height = r.height;
                        light.transform = obj.transform;
                        light.prim_path = std::move(obj.prim_path);

                        // Run property adapters on the Light
                        for (const auto* prop : k_property_adapters()) {
                            if (prop->apply(prim, light, world) == AdapterAction::Skip) return;
                        }

                        auto light_path = light.prim_path;
                        world.lights.push_back(std::move(light));
                        world.prim_to_light[std::move(light_path)] =
                            static_cast<uint32_t>(world.lights.size() - 1);
                    }
                },
                *result);

            break;  // first matching adapter wins
        }
    }
}

}  // namespace pts::rendering
