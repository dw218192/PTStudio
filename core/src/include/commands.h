#pragma once
#include <variant>
#include <glm/vec3.hpp>

#include "renderConfig.h"

/**
 * \brief A render config change command
 * This will cause the renderer to update the render setting to reflect the changes
*/
struct Cmd_ChangeRenderConfig {
    RenderConfig config;
};

using Cmd = std::variant<
    Cmd_ChangeRenderConfig
>;