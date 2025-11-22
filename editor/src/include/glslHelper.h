#pragma once
#include <string>
#include <string_view>
#include <vector>

namespace PTS {
namespace GLSLHelper {
auto preprocess(std::string_view src) -> std::string;
}
}  // namespace PTS
