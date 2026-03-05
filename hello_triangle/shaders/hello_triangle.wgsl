struct Uniforms { mvp: mat4x4f };
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
};
struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

@vertex fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = u.mvp * vec4f(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment fn fs_main(@location(0) color: vec3f) -> @location(0) vec4f {
    return vec4f(color, 1.0);
}
