import brdf;

// ---------------------------------------------------------------------------
// Hammersley quasi-random sequence
// ---------------------------------------------------------------------------

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint i, uint N) {
    return float2(float(i) / float(N), radical_inverse_vdc(i));
}

// ---------------------------------------------------------------------------
// GGX importance sampling
// ---------------------------------------------------------------------------

float3 importance_sample_ggx(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;

    float phi = 2.0 * PI * Xi.x;
    float cos_theta = sqrt((1.0 - Xi.y) / (1.0 + (a2 - 1.0) * Xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    float3 H = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    return normalize(T * H.x + B * H.y + N * H.z);
}

// ---------------------------------------------------------------------------
// Cubemap direction mapping
// ---------------------------------------------------------------------------

float3 cube_dir_from_face_uv(uint face, float2 uv) {
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;

    float3 dir;
    switch (face) {
        case 0: dir = float3( 1, -v, -u); break;  // +X
        case 1: dir = float3(-1, -v,  u); break;  // -X
        case 2: dir = float3( u,  1,  v); break;  // +Y
        case 3: dir = float3( u, -1, -v); break;  // -Y
        case 4: dir = float3( u, -v,  1); break;  // +Z
        default: dir = float3(-u, -v, -1); break;  // -Z
    }
    return normalize(dir);
}

// ---------------------------------------------------------------------------
// Equirectangular UV mapping
// ---------------------------------------------------------------------------

float2 equirect_uv_from_dir(float3 dir) {
    float phi = atan2(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return float2(phi / (2.0 * PI) + 0.5, theta / PI + 0.5);
}
