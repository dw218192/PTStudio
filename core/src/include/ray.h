#pragma once
#include <glm/glm.hpp>
#include "utils.h"

namespace PTS {
    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;

        HOST DEVICE Ray(glm::vec3 origin, glm::vec3 direction) noexcept
            : origin{ origin }, direction{ glm::normalize(direction) } { }

        HOST DEVICE auto get_point(float t) const noexcept -> glm::vec3 {
            return origin + t * direction;
        }
    };
}