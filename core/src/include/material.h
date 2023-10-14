#pragma once
#include <glm/glm.hpp>

struct Material {
    // metallic-roughness workflow
	glm::vec3 albedo { 1.0f };
    float roughness { 0.5f };
    float metallic { 0.0f };
    float ior { 1.0f };
    float ao { 1.0f };
    float transmission { 0.0f };

    struct {
        glm::vec3 color { 0.0f };
        float intensity { 0.0f };
    } emission;

    bool has_emission() const noexcept {
        return glm::length(emission.color) > 0.0f && emission.intensity > 0.0f;
    }
};