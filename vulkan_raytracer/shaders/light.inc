#ifndef LIGHT_INC
#define LIGHT_INC

#include "bindings.inc"
#include <light.inc>

layout (set = LIGHTS_SET, binding = LIGHTS_BINDING)
uniform LightBlock {
    int num_lights;
    LightData lights[MAX_LIGHTS];
};

#endif