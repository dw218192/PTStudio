#pragma once
#include <glm/glm.hpp>
#include <string_view>

namespace PTS {
	enum class LightType {
		Directional,
		Point,
		Spot,
		Mesh,
		// Object with emissive material
	};

	// used in glsl
	struct LightData {
		glm::vec3 color; // 0    base alignment: 16
		float intensity; // 12   base alignment: 4
		glm::mat4 transform; // 16   base alignment: 16
		int type; // 80   base alignment: 4
		glm::vec3 _pad; // 84
		// total size: 96

		static constexpr auto glsl_def = std::string_view{
			"struct LightData {\n"
			"   vec3 color;\n"
			"   float intensity;\n"
			"   mat4 transform;\n"
			"   int type;\n"
			"};\n"
		};
	};

	static_assert(sizeof(LightData) == 96, "LightData size mismatch");
}
