#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.inc"
#include "perframe_data.inc"

layout(location = 0)
rayPayloadInEXT Payload payload;

layout(location = 1)
rayPayloadEXT bool vis_test;

void main() {
    payload.Li = vec3(0.0);
    payload.done = true;
    vis_test = false;
}