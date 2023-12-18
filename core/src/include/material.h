#pragma once
#include <glm/glm.hpp>
#include <limits>
#include "reflection.h"

namespace PTS {
	struct Material {
		// metallic-roughness workflow
		BEGIN_REFLECT(Material, void);
		FIELD(glm::vec3, albedo, glm::vec3{ 1.0f },
		      MSerialize{}, MColor{});

		FIELD(float, roughness, 0.5f,
		      MSerialize{});

		FIELD(float, metallic, 0.0f,
		      MSerialize{});

		FIELD(float, ior, 1.0f,
		      MSerialize{});

		FIELD(float, ao, 1.0f,
		      MSerialize{});

		FIELD(float, transmission, 0.0f,
		      MSerialize{});

		FIELD(glm::vec3, emission, glm::vec3{ 0.0f },
		      MSerialize{}, MColor{});

		FIELD(float, emission_intensity, 0.0f,
		      MSerialize{}, MRange{ 0.0f, 100.0f });

		END_REFLECT();

		auto operator==(Material const& other) const noexcept -> bool {
#define FLOAT_EQ(a, b) (std::abs((a) - (b)) <= std::numeric_limits<float>::epsilon())

			return albedo == other.albedo
				&& FLOAT_EQ(roughness, other.roughness)
				&& FLOAT_EQ(metallic, other.metallic)
				&& FLOAT_EQ(ior, other.ior)
				&& FLOAT_EQ(ao, other.ao)
				&& FLOAT_EQ(transmission, other.transmission)
				&& emission == other.emission;

#undef FLOAT_EQ
		}

		auto operator!=(Material const& other) const noexcept -> bool {
			return !(*this == other);
		}

		auto is_emissive() const noexcept -> bool {
			return emission_intensity > 0.0f;
		}
	};
}
