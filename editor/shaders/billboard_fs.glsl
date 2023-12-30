#version 330 core
uniform vec3 u_tint;
uniform sampler2D u_spriteTexture;
in vec2 uv;
out vec4 FragColor;
void main() {
    FragColor = texture(u_spriteTexture, uv) * vec4(u_tint, 1.0);
}