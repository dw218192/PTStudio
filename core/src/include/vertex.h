#pragma once
#include <glm/glm.hpp>
#include "utils.h"
#include "reflection.h"

struct Vertex {
BEGIN_REFLECT(Vertex);
	FIELD_MOD(glm::vec3, position,
		MSerialize{});
	FIELD_MOD(glm::vec3, normal,
		MSerialize{});
	FIELD_MOD(glm::vec2, uv,
		MSerialize{});
END_REFLECT();
};