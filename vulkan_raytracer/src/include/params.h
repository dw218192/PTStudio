#pragma once
#include <core/consts.h>

#include <glm/glm.hpp>

namespace PTS {
namespace Vk {
static constexpr int k_max_width = 4000;   // max width of the output image
static constexpr int k_max_height = 4000;  // max height of the output image

static constexpr auto k_rgen_shader_path = "shaders/rgen.glsl";
static constexpr auto k_miss_shader_path = "shaders/miss.glsl";
static constexpr auto k_chit_shader_path = "shaders/chit.glsl";
}  // namespace Vk
}  // namespace PTS
