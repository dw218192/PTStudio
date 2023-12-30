#version 330 core
uniform sampler2D screenTexture;
uniform float thickness;
uniform vec3 outlineColor;
uniform vec2 texelSize;
in vec2 TexCoords;
out vec4 FragColor;
void main() {
    const vec3 target = vec3(0.0, 0.0, 0.0); // Find black 
    const float TAU = 6.28318530;
    const float steps = 32.0;
    if (texture(screenTexture, TexCoords).r > 0) {
        FragColor.a = 0.0;
        return;
    }
    for (float i = 0.0; i < TAU; i += TAU / steps) {
        // Sample image in a circular pattern
        vec2 offset = vec2(sin(i), cos(i)) * texelSize * thickness;
        vec4 col = texture(screenTexture, TexCoords + offset);
        float alpha = smoothstep(0.5, 0.7, distance(col.rgb, target));
        FragColor = mix(FragColor, vec4(outlineColor, 1.0), alpha);
    }
    vec4 mat = texture(screenTexture, TexCoords);
    float factor = smoothstep(0.5, 0.7, distance(mat.rgb, target));
    FragColor = mix(FragColor, mat, factor);
}