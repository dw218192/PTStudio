#ifndef CORE_LIGHT_INC
#define CORE_LIGHT_INC

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

#endif // CORE_LIGHT_INC