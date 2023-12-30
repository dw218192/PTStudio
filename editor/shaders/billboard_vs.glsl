#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;
uniform float u_scale;
uniform vec3 u_worldPos;
uniform mat4 u_view;
uniform mat4 u_projection;
out vec2 uv;
void main() {
    vec3 pos = vec3(u_view * vec4(u_worldPos, 1.0));
    vec3 n = -normalize(pos);
    vec3 u = normalize(cross(vec3(0.0, 1.0, 0.0), n));
    vec3 v = normalize(cross(n, u));
    vec3 objPos = aPos * u_scale;
    pos = objPos.x * u + objPos.y * v + objPos.z * n + pos;
    gl_Position = u_projection * vec4(pos, 1.0);
    uv = aTexCoords;
}