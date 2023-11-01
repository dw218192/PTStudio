#pragma once
#include <glm/glm.hpp>
#include "utils.h"
#include "reflection.h"

struct Vertex {
BEGIN_REFLECT(Vertex);
    FIELD(glm::vec3, position);
    FIELD(glm::vec3, normal);
    FIELD(glm::vec2, uv);
END_REFLECT();
};