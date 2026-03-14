#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <pxr/usd/usdGeom/sphere.h>

#include <cmath>

namespace pts::rendering {

namespace {
constexpr int k_lat_segments = 16;
constexpr int k_lon_segments = 32;
constexpr float k_pi = 3.14159265358979323846f;
}  // namespace

const SphereAdapter& SphereAdapter::instance() {
    static const SphereAdapter s_instance;
    return s_instance;
}

bool SphereAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomSphere>();
}

std::optional<AdapterResult> SphereAdapter::adapt(const pxr::UsdPrim& prim) const {
    pxr::UsdGeomSphere sphere(prim);

    double radius = 1.0;
    sphere.GetRadiusAttr().Get(&radius);
    float r = static_cast<float>(radius);

    auto colors = read_display_color(prim);

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Generate vertices: (k_lat_segments + 1) rings x (k_lon_segments + 1) columns
    int rows = k_lat_segments + 1;
    int cols = k_lon_segments + 1;
    vertices.reserve(rows * cols);

    for (int lat = 0; lat <= k_lat_segments; ++lat) {
        float theta = static_cast<float>(lat) / k_lat_segments * k_pi;
        float sin_theta = std::sin(theta);
        float cos_theta = std::cos(theta);
        float v_coord = static_cast<float>(lat) / k_lat_segments;

        for (int lon = 0; lon <= k_lon_segments; ++lon) {
            float phi = static_cast<float>(lon) / k_lon_segments * 2.0f * k_pi;
            float sin_phi = std::sin(phi);
            float cos_phi = std::cos(phi);

            float nx = sin_theta * cos_phi;
            float ny = cos_theta;
            float nz = sin_theta * sin_phi;

            Vertex vtx = {};
            vtx.position[0] = r * nx;
            vtx.position[1] = r * ny;
            vtx.position[2] = r * nz;
            vtx.normal[0] = nx;
            vtx.normal[1] = ny;
            vtx.normal[2] = nz;
            vtx.uv[0] = static_cast<float>(lon) / k_lon_segments;
            vtx.uv[1] = v_coord;
            apply_display_color(vtx, colors);
            vertices.push_back(vtx);
        }
    }

    // Generate indices
    indices.reserve(k_lat_segments * k_lon_segments * 6);
    for (int lat = 0; lat < k_lat_segments; ++lat) {
        for (int lon = 0; lon < k_lon_segments; ++lon) {
            uint32_t a = static_cast<uint32_t>(lat * cols + lon);
            uint32_t b = a + 1;
            uint32_t c = static_cast<uint32_t>((lat + 1) * cols + lon);
            uint32_t d = c + 1;

            // Skip degenerate triangles at poles
            if (lat != 0) {
                indices.push_back(a);
                indices.push_back(c);
                indices.push_back(b);
            }
            if (lat != k_lat_segments - 1) {
                indices.push_back(b);
                indices.push_back(c);
                indices.push_back(d);
            }
        }
    }

    return AdapterResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
