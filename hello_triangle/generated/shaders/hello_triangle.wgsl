struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct Uniforms_std140_0
{
    @align(16) mvp_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
    @align(16) time_0 : f32,
    @align(4) rotation_0 : f32,
};

@binding(0) @group(0) var<uniform> u_0 : Uniforms_std140_0;
struct VsOut_0
{
    @builtin(position) position_0 : vec4<f32>,
    @location(0) color_0 : vec3<f32>,
};

struct vertexInput_0
{
    @location(0) position_1 : vec3<f32>,
    @location(1) normal_0 : vec3<f32>,
    @location(2) color_1 : vec3<f32>,
};

@vertex
fn vs_main( _S1 : vertexInput_0) -> VsOut_0
{
    var c_0 : f32 = cos(u_0.rotation_0);
    var s_0 : f32 = sin(u_0.rotation_0);
    var _S2 : f32 = _S1.position_1.x;
    var _S3 : f32 = _S1.position_1.z;
    var output_0 : VsOut_0;
    output_0.position_0 = (((vec4<f32>(vec3<f32>(_S2 * c_0 + _S3 * s_0, _S1.position_1.y, - _S2 * s_0 + _S3 * c_0), 1.0f)) * (mat4x4<f32>(u_0.mvp_0.data_0[i32(0)][i32(0)], u_0.mvp_0.data_0[i32(1)][i32(0)], u_0.mvp_0.data_0[i32(2)][i32(0)], u_0.mvp_0.data_0[i32(3)][i32(0)], u_0.mvp_0.data_0[i32(0)][i32(1)], u_0.mvp_0.data_0[i32(1)][i32(1)], u_0.mvp_0.data_0[i32(2)][i32(1)], u_0.mvp_0.data_0[i32(3)][i32(1)], u_0.mvp_0.data_0[i32(0)][i32(2)], u_0.mvp_0.data_0[i32(1)][i32(2)], u_0.mvp_0.data_0[i32(2)][i32(2)], u_0.mvp_0.data_0[i32(3)][i32(2)], u_0.mvp_0.data_0[i32(0)][i32(3)], u_0.mvp_0.data_0[i32(1)][i32(3)], u_0.mvp_0.data_0[i32(2)][i32(3)], u_0.mvp_0.data_0[i32(3)][i32(3)]))));
    var phase_0 : f32 = u_0.time_0 * 0.5f;
    var _S4 : vec3<f32> = vec3<f32>(0.5f);
    output_0.color_0 = _S1.color_1 * _S4 + _S4 * vec3<f32>(sin(phase_0) * 0.5f + 0.5f, sin(phase_0 + 2.09400010108947754f) * 0.5f + 0.5f, sin(phase_0 + 4.18900012969970703f) * 0.5f + 0.5f);
    return output_0;
}

struct pixelOutput_0
{
    @location(0) output_1 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) color_2 : vec3<f32>,
};

@fragment
fn fs_main( _S5 : pixelInput_0) -> pixelOutput_0
{
    var _S6 : pixelOutput_0 = pixelOutput_0( vec4<f32>(_S5.color_2, 1.0f) );
    return _S6;
}

