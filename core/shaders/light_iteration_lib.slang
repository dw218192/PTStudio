// Brute-force light iteration module.
// V1: iterates all lights. When clustered lighting lands, only this module
// changes (cluster lookup from screen_uv + depth).

public uint get_light_count(uint total_light_count, float2 screen_uv) {
    return total_light_count;
}

public uint get_light_index(uint i, float2 screen_uv) {
    return i;
}
