#pragma once
#include <string>
#include <vector>
#include <string_view>
#include "shaderType.h"

namespace PTS {
	namespace GLSLHelper {
		// shader code analysis (fake)
		struct GLSLInfo {
			std::string inputs, outputs;
		};

		auto preprocess(ShaderType type, std::string_view common_funcs, std::string_view main_src) -> std::string;
		auto get_in_out(ShaderType type, std::string_view src) -> GLSLInfo;
	}
}
