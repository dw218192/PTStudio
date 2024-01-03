#version 420 core

#include "uniforms.inc"
#include "light.inc"
#include "bindings.inc"

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;
out vec4 FragColor;

void main() {
    vec3 camPos = u_view[3].xyz;
    vec3 result = vec3(0.0);
    for (int i=0; i<u_lightCount; ++i) {
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(u_lights[i].transform[3].xyz - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * u_lights[i].color;
        float specularStrength = 0.5;
        vec3 viewDir = normalize(camPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * u_lights[i].color;

        // attenuation
        float distance = length(u_lights[i].transform[3].xyz - FragPos);
        float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
        result += (diffuse + specular) * u_lights[i].intensity * attenuation * u_objectColor;
    }
    // ambient
    result += vec3(0.2);
    FragColor = vec4(result, 1.0);
}