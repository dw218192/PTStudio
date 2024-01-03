#ifndef MESH_ATTRS_INC
#define MESH_ATTRS_INC

#include "bindings.inc"

#include <consts.inc>
#include <material.inc>

struct VertexAttribData {
    vec3 normal;
    vec2 tex_coord;
};

layout (set = VERTEX_ATTR_SET, binding = VERTEX_ATTR_BINDING) 
readonly buffer VertexAttribsBlock {
    VertexAttribData data[];
} vertexAttribs[];

layout (set = INDICES_SET, binding = INDICES_BINDING)
readonly buffer FaceIndicesBlock {
    uvec3 data[];
} faceIndices[];

layout (set = MATERIAL_SET, binding = MATERIAL_BINDING)
uniform MaterialBlock {
    MaterialData materials[MAX_INSTANCES];
};

#endif