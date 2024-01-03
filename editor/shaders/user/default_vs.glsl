#version 420 core

#include "uniforms.inc"
#include "light.inc"
#include "bindings.inc"

layout (location = VERTEX_POS_BINDING)
in vec3 aPos;

layout (location = VERTEX_NORMAL_BINDING)
in vec3 aNormal;

layout (location = VERTEX_UV_BINDING)
in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

void main() {
    TexCoords = aTexCoords;
    Normal = mat3(transpose(inverse(u_model))) * aNormal;
    FragPos = vec3(u_model * vec4(aPos, 1.0));
    gl_Position = u_projection * u_view * vec4(FragPos, 1.0);
}