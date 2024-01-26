#pragma once
#include <string>
#include <vector>
#include <string_view>

namespace PTS {
	namespace GLSLHelper {
		auto preprocess(std::string_view src) -> std::string;
	}
}
