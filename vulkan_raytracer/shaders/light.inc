#ifndef LIGHT_INC
#define LIGHT_INC

#include "bindings.inc"

const int DIRECTIONAL_LIGHT = 0;
const int POINT_LIGHT = 1;
const int SPOT_LIGHT = 2;
const int MESH_LIGHT = 3;

struct LightData {
   vec3 color;
   float intensity;
   mat4 transform;
   int type;
};

layout (set = LIGHTS_SET, binding = LIGHTS_BINDING)
uniform LightBlock {
    int num_lights;
    LightData lights[MAX_LIGHTS];
};

#endif