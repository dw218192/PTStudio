#pragma once
#include <glm/glm.hpp>
#include "reflection.h"
namespace PTS {
    struct Material {
        // metallic-roughness workflow
        BEGIN_REFLECT(Material);
        FIELD_MOD(glm::vec3, albedo, glm::vec3{ 1.0f },
            MSerialize{}, MColor{});
        FIELD_MOD(float, roughness, 0.5f,
            MSerialize{});
        FIELD_MOD(float, metallic, 0.0f,
            MSerialize{});
        FIELD_MOD(float, ior, 1.0f,
            MSerialize{});
        FIELD_MOD(float, ao, 1.0f,
            MSerialize{});
        FIELD_MOD(float, transmission, 0.0f,
            MSerialize{});
        FIELD_MOD(glm::vec3, emission, glm::vec3{ 0.0f },
            MSerialize{}, MColor{});
        FIELD_MOD(float, emission_intensity, 1.0f,
            MSerialize{});
        END_REFLECT();

        bool operator==(Material const& other) const noexcept {
            return albedo == other.albedo
                && roughness == other.roughness
                && metallic == other.metallic
                && ior == other.ior
                && ao == other.ao
                && transmission == other.transmission
                && emission == other.emission;
        }
        bool operator!=(Material const& other) const noexcept {
            return !(*this == other);
        }
        bool is_emissive() const noexcept {
            return glm::length(emission) > 0.0f;
        }
    };
}