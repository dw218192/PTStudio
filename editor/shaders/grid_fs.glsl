#version 330 core
uniform float halfGridDim;
in vec2 gridCoords;
out vec4 FragColor;
void main() {
    float dist = max(abs(gridCoords.x), abs(gridCoords.y)) / halfGridDim;
    float alpha = 1.0 - pow(dist, 0.25);
    FragColor = vec4(0.7, 0.7, 0.7, alpha);
}