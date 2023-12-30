#ifndef PERFRAME_DATA_INC
#define PERFRAME_DATA_INC

struct CameraData {
    mat4 inv_view_proj;
    vec3 pos;
};
struct PerFrameData {
    CameraData camera;
    int iteration;
    int num_samples;
    int max_bounces;
    bool direct_lighting_only;
};
layout (push_constant) uniform PerFrameDataBlock {
    PerFrameData perFrameData;
};

#endif