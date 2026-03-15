#include <core/diagnostics.h>
#include <core/rendering/wireframeIndices.h>

namespace pts::rendering {

std::vector<uint32_t> expand_wireframe_indices(const uint32_t* tri_indices,
                                               size_t tri_index_count) {
    PRECONDITION_MSG(tri_index_count % 3 == 0, "Triangle index count must be a multiple of 3");

    std::vector<uint32_t> lines;
    lines.reserve(tri_index_count * 2);
    for (size_t i = 0; i + 2 < tri_index_count; i += 3) {
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
