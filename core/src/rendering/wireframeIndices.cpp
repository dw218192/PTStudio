#include <core/diagnostics.h>
#include <core/rendering/wireframeIndices.h>

namespace pts::rendering {

std::vector<uint32_t> expand_wireframe_indices(const std::vector<uint32_t>& tri_indices) {
    PRECONDITION_MSG(tri_indices.size() % 3 == 0, "Triangle index count must be a multiple of 3");

    std::vector<uint32_t> lines;
    lines.reserve(tri_indices.size() * 2);
    for (size_t i = 0; i + 2 < tri_indices.size(); i += 3) {
        auto a = tri_indices[i];
        auto b = tri_indices[i + 1];
        auto c = tri_indices[i + 2];
        lines.push_back(a);
        lines.push_back(b);
        lines.push_back(b);
        lines.push_back(c);
        lines.push_back(c);
        lines.push_back(a);
    }
    return lines;
}

}  // namespace pts::rendering
