#ifndef CORE_MATERIAL_DATA_H
#define CORE_MATERIAL_DATA_H

const int DIFFUSE = 0;
const int PERFECT_SPECULAR = 1;


struct MaterialData {
    vec3 base_color;
    vec3 emissive_color;
    float emission_intensity;
};

int get_material_type(in MaterialData material) {
    return DIFFUSE;
}

#endif // CORE_MATERIAL_DATA_H