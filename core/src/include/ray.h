#pragma once
#include <glm/glm.hpp>
namespace PTS {
    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;

        Ray(glm::vec3 origin, glm::vec3 direction) noexcept
            : origin{ origin }, direction{ glm::normalize(direction) } { }

        glm::vec3 get_point(float t) const noexcept {
            return origin + t * direction;
        }
    };
}