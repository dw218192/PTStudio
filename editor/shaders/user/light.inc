#ifndef EDITOR_LIGHT_INC
#define EDITOR_LIGHT_INC

#include <consts.inc>
#include <light.inc>
#include "bindings.inc"

layout (std140, binding = LIGHT_UBO_BINDING)
uniform LightBlock {
    LightData u_lights[MAX_LIGHTS];
};

#endif