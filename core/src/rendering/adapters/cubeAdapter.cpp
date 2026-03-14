#include <core/rendering/adapters/adapterUtils.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <pxr/usd/usdGeom/cube.h>

#include <cmath>

namespace pts::rendering {

const CubeAdapter& CubeAdapter::instance() {
    static const CubeAdapter s_instance;
    return s_instance;
}

bool CubeAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCube>();
}

std::optional<AdapterResult> CubeAdapter::adapt(const pxr::UsdPrim& prim) const {
    pxr::UsdGeomCube cube(prim);

    double size = 2.0;
    cube.GetSizeAttr().Get(&size);
    float h = static_cast<float>(size) * 0.5f;

    auto colors = read_display_color(prim);

    // 24 vertices (4 per face, 6 faces) with per-face normals
    // Face order: +X, -X, +Y, -Y, +Z, -Z
    struct FaceDef {
        float normal[3];
        float positions[4][3];
        float uvs[4][2];
    };

    // clang-format off
    const FaceDef faces[6] = {
        // +X
        {{ 1, 0, 0}, {{ h,-h,-h}, { h,-h, h}, { h, h, h}, { h, h,-h}}, {{0,0},{1,0},{1,1},{0,1}}},
        // -X
        {{-1, 0, 0}, {{-h,-h, h}, {-h,-h,-h}, {-h, h,-h}, {-h, h, h}}, {{0,0},{1,0},{1,1},{0,1}}},
        // +Y
        {{ 0, 1, 0}, {{-h, h,-h}, {-h, h, h}, { h, h, h}, { h, h,-h}}, {{0,0},{1,0},{1,1},{0,1}}},
        // -Y
        {{ 0,-1, 0}, {{-h,-h, h}, {-h,-h,-h}, { h,-h,-h}, { h,-h, h}}, {{0,0},{1,0},{1,1},{0,1}}},
        // +Z
        {{ 0, 0, 1}, {{-h,-h, h}, { h,-h, h}, { h, h, h}, {-h, h, h}}, {{0,0},{1,0},{1,1},{0,1}}},
        // -Z
        {{ 0, 0,-1}, {{ h,-h,-h}, {-h,-h,-h}, {-h, h,-h}, { h, h,-h}}, {{0,0},{1,0},{1,1},{0,1}}},
    };
    // clang-format on

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(24);
    indices.reserve(36);

    for (const auto& face : faces) {
        auto base = static_cast<uint32_t>(vertices.size());
        for (int j = 0; j < 4; ++j) {
            Vertex v = {};
            v.position[0] = face.positions[j][0];
            v.position[1] = face.positions[j][1];
            v.position[2] = face.positions[j][2];
            v.normal[0] = face.normal[0];
            v.normal[1] = face.normal[1];
            v.normal[2] = face.normal[2];
            v.uv[0] = face.uvs[j][0];
            v.uv[1] = face.uvs[j][1];
            apply_display_color(v, colors);
            vertices.push_back(v);
        }
        // Two triangles per quad
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }

    return AdapterResult{std::move(vertices), std::move(indices)};
}

}  // namespace pts::rendering
