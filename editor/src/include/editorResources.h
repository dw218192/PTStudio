#pragma once

constexpr char const* k_imgui_ini = 
"\n\
[Window][Debug##Default]\n\
ViewportPos=10,10\n\
ViewportId=0x9F5F46A1\n\
Size=400,400\n\
Collapsed=0\n\
\n\
[Window][Scene Settings]\n\
Pos=0,0\n\
Size=187,542\n\
Collapsed=0\n\
DockId=0x00000002,0\n\
\n\
[Window][DockSpaceViewport_11111111]\n\
Pos=0,0\n\
Size=1280,720\n\
Collapsed=0\n\
\n\
[Window][Scene]\n\
Pos=189,0\n\
Size=880,542\n\
Collapsed=0\n\
DockId=0x00000003,0\n\
\n\
[Window][Inspector]\n\
Pos=1071,0\n\
Size=209,542\n\
Collapsed=0\n\
DockId=0x00000004,0\n\
\n\
[Window][Console]\n\
Pos=0,544\n\
Size=1280,176\n\
Collapsed=0\n\
DockId=0x00000006,0\n\
\n\
[Docking][Data]\n\
DockSpace       ID=0x8B93E3BD Window=0xA787BDB4 Pos=164,187 Size=1280,720 Split=Y Selected=0xE192E354\n\
DockNode      ID=0x00000005 Parent=0x8B93E3BD SizeRef=1280,542 Split=X\n\
    DockNode    ID=0x00000001 Parent=0x00000005 SizeRef=1069,720 Split=X\n\
    DockNode  ID=0x00000002 Parent=0x00000001 SizeRef=187,720 Selected=0xD4E24632\n\
    DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=880,720 CentralNode=1 Selected=0xE192E354\n\
    DockNode    ID=0x00000004 Parent=0x00000005 SizeRef=209,720 Selected=0xE7039252\n\
DockNode      ID=0x00000006 Parent=0x8B93E3BD SizeRef=1280,176 Selected=0x49278EEE\n\
\n\
";

// shaders

constexpr char const* vs_obj_src =
"\
#version 330 core\n\
layout (location = 0) in vec3 aPos;\n\
layout (location = 1) in vec3 aNormal;\n\
layout (location = 2) in vec2 aTexCoords;\n\
out vec2 TexCoords;\n\
out vec3 Normal;\n\
out vec3 FragPos;\n\
uniform mat4 model;\n\
uniform mat4 view;\n\
uniform mat4 projection;\n\
void main() {\n\
    TexCoords = aTexCoords;\n\
    Normal = mat3(transpose(inverse(model))) * aNormal;\n\
    FragPos = vec3(model * vec4(aPos, 1.0));\n\
    gl_Position = projection * view * vec4(FragPos, 1.0);\n\
}\n\
";

constexpr char const* ps_obj_src =
"\
#version 330 core\n\
out vec4 FragColor;\n\
in vec2 TexCoords;\n\
in vec3 Normal;\n\
in vec3 FragPos;\n\
uniform vec3 lightPos;\n\
uniform mat4 view;\n\
void main() {\n\
    const vec3 objectColor = vec3(178.0/255.0, 190.0/255.0, 181.0/255.0);\n\
    const vec3 lightColor = vec3(1.0, 1.0, 1.0);\n\
    vec3 camPos = view[3].xyz;\n\
    float ambientStrength = 0.2;\n\
    vec3 ambient = ambientStrength * lightColor;\n\
    vec3 norm = normalize(Normal);\n\
    vec3 lightDir = normalize(lightPos - FragPos);\n\
    float diff = max(dot(norm, lightDir), 0.0);\n\
    vec3 diffuse = diff * lightColor;\n\
    float specularStrength = 0.5;\n\
    vec3 viewDir = normalize(camPos - FragPos);\n\
    vec3 reflectDir = reflect(-lightDir, norm);\n\
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);\n\
    vec3 specular = specularStrength * spec * lightColor;\n\
    vec3 result = (ambient + diffuse + specular) * objectColor;\n\
    FragColor = vec4(result, 1.0);\n\
}\n\
";

// outline works by first rendering the object slightly scaled up, with a solid color
// then rendering the object normally, with the outline color, but with depth testing disabled
constexpr char const* vs_outline_passes[] = {
    "\
    #version 330 core\n\
    layout (location = 0) in vec3 aPos;\n\
    uniform mat4 model;\n\
    uniform mat4 view;\n\
    uniform mat4 projection;\n\
    void main() {\n\
        gl_Position = projection * view * model * vec4(aPos, 1.0);\n\
    }\n\
    ",
    // post-process pass 1 & 2 & 3, draw full screen quad
    "\
    #version 330 core\n\
    layout (location = 0) in vec3 aPos;\n\
    layout (location = 1) in vec2 aTexCoords;\n\
    out vec2 TexCoords;\n\
    void main() {\n\
        TexCoords = aTexCoords;\n\
        gl_Position = vec4(aPos, 1.0);\n\
    }\n\
    ",
    "\
    #version 330 core\n\
    layout (location = 0) in vec3 aPos;\n\
    layout (location = 1) in vec2 aTexCoords;\n\
    out vec2 TexCoords;\n\
    void main() {\n\
        TexCoords = aTexCoords;\n\
        gl_Position = vec4(aPos, 1.0);\n\
    }\n\
    ",
};
constexpr char const* ps_outline_passes[] = {
    "\
    #version 330 core\n\
    layout(location = 0) out vec3 FragColor; \n\
    void main() {\n\
        FragColor = vec3(1.0);\n\
    }\n\
    ",
    // post-process pass, screen-space outline from shadertoy
    "\
    #version 330 core\n\
    uniform sampler2D screenTexture;\n\
    uniform float thickness;\n\
    uniform vec3 outlineColor;\n\
    uniform vec2 texelSize;\n\
    in vec2 TexCoords;\n\
    out vec4 FragColor;\n\
    void main() {\n\
        const vec3 target = vec3(0.0, 0.0, 0.0); // Find black \n\
        const float TAU = 6.28318530;\n\
        const float steps = 32.0;\n\
        if (texture(screenTexture, TexCoords).r > 0) {\n\
            FragColor.a = 0.0;\n\
            return;\n\
        }\n\
        for (float i = 0.0; i < TAU; i += TAU / steps) {\n\
            // Sample image in a circular pattern\n\
            vec2 offset = vec2(sin(i), cos(i)) * texelSize * thickness;\n\
            vec4 col = texture(screenTexture, TexCoords + offset);\n\
            float alpha = smoothstep(0.5, 0.7, distance(col.rgb, target));\n\
            FragColor = mix(FragColor, vec4(outlineColor, 1.0), alpha);\n\
        }\n\
        vec4 mat = texture(screenTexture, TexCoords);\n\
        float factor = smoothstep(0.5, 0.7, distance(mat.rgb, target));\n\
        FragColor = mix(FragColor, mat, factor);\n\
    }\n\
    ",
    // copy screen texture to screen
    "\
    #version 330 core\n\
    uniform sampler2D screenTexture;\n\
    in vec2 TexCoords;\n\
    out vec4 FragColor;\n\
    void main() {\n\
        FragColor = texture(screenTexture, TexCoords);\n\
    }\n\
    ",
};

constexpr char const* vs_grid_src = 
"\
#version 330 core\n\
layout (location = 0) in vec3 aPos;\n\
uniform mat4 view;\n\
uniform mat4 projection;\n\
out vec2 grid_coords;\n\
void main() {\n\
    grid_coords = aPos.xz;\n\
    gl_Position = projection * view * vec4(aPos, 1.0);\n\
}\n\
";

constexpr char const* ps_grid_src = 
"\
#version 330 core\n\
uniform float half_grid_dim;\n\
in vec2 grid_coords;\n\
out vec4 FragColor;\n\
void main() {\n\
    float dist = max(abs(grid_coords.x), abs(grid_coords.y)) / half_grid_dim;\n\
    float alpha = 1.0 - pow(dist, 0.25);\n\
    FragColor = vec4(0.7, 0.7, 0.7, alpha);\n\
}\n\
";

constexpr char const* k_uniform_model = "model";
constexpr char const* k_uniform_view = "view";
constexpr char const* k_uniform_projection = "projection";
constexpr char const* k_uniform_light_pos = "lightPos";
constexpr char const* k_uniform_light_color = "lightColor";
constexpr char const* k_uniform_object_color = "objectColor";
constexpr char const* k_uniform_half_grid_dim = "half_grid_dim";


constexpr char const* k_uniform_screen_texture = "screenTexture";
constexpr char const* k_uniform_outline_color = "outlineColor";
constexpr char const* k_uniform_texel_size = "texelSize";
constexpr char const* k_uniform_thickness = "thickness";


// for shader editor https://github.com/BalazsJako/ImGuiColorTextEdit/issues/121
static const char* const glsl_keywords[] = {
    "const", "uniform", "buffer", "shared", "attribute", "varying",
    "coherent", "volatile", "restrict", "readonly", "writeonly",
    "atomic_uint",
    "layout",
    "centroid", "flat", "smooth", "noperspective",
    "patch", "sample",
    "invariant", "precise",
    "break", "continue", "do", "for", "while", "switch", "case", "default",
    "if", "else",
    "subroutine",
    "in", "out", "inout",
    "int", "void", "bool", "true", "false", "float", "double",
    "discard", "return",
    "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4",
    "uint", "uvec2", "uvec3", "uvec4",
    "dvec2", "dvec3", "dvec4",
    "mat2", "mat3", "mat4",
    "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4",
    "mat4x2", "mat4x3", "mat4x4",
    "dmat2", "dmat3", "dmat4",
    "dmat2x2", "dmat2x3", "dmat2x4",
    "dmat3x2", "dmat3x3", "dmat3x4",
    "dmat4x2", "dmat4x3", "dmat4x4",
    "lowp", "mediump", "highp", "precision",
    "sampler1D", "sampler1DShadow", "sampler1DArray", "sampler1DArrayShadow",
    "isampler1D", "isampler1DArray", "usampler1D usampler1DArray",
    "sampler2D", "sampler2DShadow", "sampler2DArray", "sampler2DArrayShadow",
    "isampler2D", "isampler2DArray", "usampler2D", "usampler2DArray",
    "sampler2DRect", "sampler2DRectShadow", "isampler2DRect", "usampler2DRect",
    "sampler2DMS", "isampler2DMS", "usampler2DMS",
    "sampler2DMSArray", "isampler2DMSArray", "usampler2DMSArray",
    "sampler3D", "isampler3D", "usampler3D",
    "samplerCube", "samplerCubeShadow", "isamplerCube", "usamplerCube",
    "samplerCubeArray", "samplerCubeArrayShadow",
    "isamplerCubeArray", "usamplerCubeArray",
    "samplerBuffer", "isamplerBuffer", "usamplerBuffer",
    "image1D", "iimage1D", "uimage1D",
    "image1DArray", "iimage1DArray", "uimage1DArray",
    "image2D", "iimage2D", "uimage2D",
    "image2DArray", "iimage2DArray", "uimage2DArray",
    "image2DRect", "iimage2DRect", "uimage2DRect",
    "image2DMS", "iimage2DMS", "uimage2DMS",
    "image2DMSArray", "iimage2DMSArray", "uimage2DMSArray",
    "image3D", "iimage3D", "uimage3D",
    "imageCube", "iimageCube", "uimageCube",
    "imageCubeArray", "iimageCubeArray", "uimageCubeArray",
    "imageBuffer", "iimageBuffer", "uimageBuffer",
    "struct"
};

static const char* const glsl_identifiers[] = {
    "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
    "abs", "sign", "floor", "trunc", "round", "roundEven", "ceil", "fract", "mod", "modf", "min", "max", "clamp", "mix", "step", "smoothstep", "isnan", "isinf", "floatBitsToInt", "floatBitsToUint", "intBitsToFloat", "uintBitsToFloat", "fma", "frexp", "ldexp",
    "packUnorm2x16", "packSnorm2x16", "packUnorm4x8", "packSnorm4x8", "unpackUnorm2x16", "unpackSnorm2x16", "unpackUnorm4x8", "unpackSnorm4x8", "packHalf2x16", "unpackHalf2x16", "packDouble2x32", "unpackDouble2x32",
    "length", "distance", "dot", "cross", "normalize", "ftransform", "faceforward", "reflect", "refract",
    "matrixCompMult", "outerProduct", "transpose", "determinant", "inverse",
    "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual", "equal", "notEqual", "any", "all", "not",
    "uaddCarry", "usubBorrow", "umulExtended", "imulExtended", "bitfieldExtract", "bitfieldInsert", "bitfieldReverse", "bitCount", "findLSB", "findMSB",
    "textureSize", "textureQueryLod", "textureQueryLevels", "textureSamples",
    "texture", "textureProj", "textureLod", "textureOffset", "texelFetch", "texelFetchOffset", "textureProjOffset", "textureLodOffset", "textureProjLod", "textureProjLodOffset", "textureGrad", "textureGradOffset", "textureProjGrad", "textureProjGradOffset",
    "textureGather", "textureGatherOffset", "textureGatherOffsets",
    "texture1D", "texture1DProj", "texture1DLod", "texture1DProjLod", "texture2D", "texture2DProj", "texture2DLod", "texture2DProjLod", "texture3D", "texture3DProj", "texture3DLod", "texture3DProjLod", "textureCube", "textureCubeLod", "shadow1D", "shadow2D", "shadow1DProj", "shadow2DProj", "shadow1DLod", "shadow2DLod", "shadow1DProjLod", "shadow2DProjLod",
    "atomicCounterIncrement", "atomicCounterDecrement", "atomicCounter", "atomicCounterAdd", "atomicCounterSubtract", "atomicCounterMin", "atomicCounterMax", "atomicCounterAnd", "atomicCounterOr", "atomicCounterXor", "atomicCounterExchange", "atomicCounterCompSwap",
    "atomicAdd", "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExchange", "atomicCompSwap",
    "imageSize", "imageSamples", "imageLoad", "imageStore", "imageAtomicAdd", "imageAtomicMin", "imageAtomicMax", "imageAtomicAnd", "imageAtomicOr", "imageAtomicXor", "imageAtomicExchange", "imageAtomicCompSwap",
    "EmitStreamVertex", "EndStreamPrimitive", "EmitVertex", "EndPrimitive",
    "dFdx", "dFdy", "dFdxFine", "dFdyFine", "dFdxCoarse", "dFdyCoarse", "fwidth", "fwidthFine", "fwidthCoarse",
    "interpolateAtCentroid", "interpolateAtSample", "interpolateAtOffset",
    "noise1", "noise2", "noise3", "noise4",
    "barrier",
    "memoryBarrier", "memoryBarrierAtomicCounter", "memoryBarrierBuffer", "memoryBarrierShared", "memoryBarrierImage", "groupMemoryBarrier",
    "subpassLoad",
    "anyInvocation", "allInvocations", "allInvocationsEqual"
};