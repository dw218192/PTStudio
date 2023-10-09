#pragma once
#include <variant>
#include "ext.h"
#include "renderConfig.h"

/**
 * \brief A camera rotation command.
 * This causes the camera to rotated by the given 3 Euler angles specified in *degrees*,
 * relative to its current rotation.
*/
struct Cmd_CameraRot {
	glm::vec3 angles_deg;
};

/**
 * \brief A camera movement command.
 * This causes the camera to move by the given offset, relative to its current position.
*/
struct Cmd_CameraMove {
	glm::vec3 delta;
};

/**
 * \brief A camera zoom command.
 * This causes the camera to zoom in/out by the given amount.
 * If the delta is positive, the camera zooms in, if it is negative, the camera zooms out.
*/
struct Cmd_CameraZoom {
    float delta;
};

/**
 * \brief A render config change command
 * This will cause the renderer to update the render setting to reflect the changes
*/
struct Cmd_ChangeRenderConfig {
    RenderConfig config;
};

using Cmd = std::variant<
    Cmd_CameraRot,
    Cmd_CameraMove,
    Cmd_CameraZoom,
    Cmd_ChangeRenderConfig
>;