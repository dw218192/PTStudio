#pragma once
#include <string>
#include <string_view>
#include <vector>

namespace pts {
namespace glsl_helper {
auto preprocess(std::string_view src) -> std::string;
}
}  // namespace pts
