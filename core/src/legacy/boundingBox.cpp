#include <core/legacy/boundingBox.h>

auto PTS::BoundingBox::from_vertices(std::vector<Vertex> const& vertices) noexcept -> BoundingBox {
    glm::vec3 min_pos{std::numeric_limits<float>::max()};
    glm::vec3 max_pos{std::numeric_limits<float>::lowest()};
    for (auto&& v : vertices) {
        min_pos = glm::min(min_pos, v.position);
        max_pos = glm::max(max_pos, v.position);
    }
    return {min_pos, max_pos};
}