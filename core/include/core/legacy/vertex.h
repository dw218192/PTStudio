#pragma once
#include <glm/glm.hpp>

#include "reflection.h"
namespace PTS {
struct Vertex {
    BEGIN_REFLECT(Vertex, void);
    FIELD(glm::vec3, position, {}, MSerialize{});
    FIELD(glm::vec3, normal, {}, MSerialize{});
    FIELD(glm::vec2, uv, {}, MSerialize{});
    END_REFLECT();
};
}  // namespace PTS