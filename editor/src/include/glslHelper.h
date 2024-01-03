#pragma once
#include <string>
#include <vector>
#include <string_view>
#include "shaderType.h"

namespace PTS {
	namespace GLSLHelper {
		auto preprocess(std::string_view src) -> std::string;
	}
}
