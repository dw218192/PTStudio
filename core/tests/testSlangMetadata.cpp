#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/rendering/shaderc/slangRuntime.h>
#include <doctest/doctest.h>
#include <slang-com-ptr.h>
#include <slang.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

using namespace pts::rendering;
namespace fs = std::filesystem;

namespace {

fs::path unique_dir(const char* tag) {
    auto base = fs::temp_directory_path() / "pts_slang_meta_test";
    auto dir = base / (std::string(tag) + "_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

fs::path write_slang(const fs::path& dir, const std::string& name, const std::string& contents) {
    auto path = dir / name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return path;
}

SlangCompileOutput compile(const fs::path& dir, const std::string& slang_src,
                           const SlangCompileOptions& options = {}) {
    auto path = write_slang(dir, "shader.slang", slang_src);
    Slang::ComPtr<slang::IGlobalSession> gs;
    REQUIRE(SLANG_SUCCEEDED(slang::createGlobalSession(gs.writeRef())));
    REQUIRE(gs);

    std::vector<std::string> entries;  // auto-discover
    std::vector<std::string_view> defines;
    std::vector<std::filesystem::path> search_paths{dir};
    return run_slang(gs.get(), search_paths, path, entries,
                     boost::span<const std::string_view>(defines.data(), defines.size()),
                     options.cpp ? "" : "test_shader", options);
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("upload types preserve padding, nested structs and array stride") {
    SlangCompileOptions options;
    options.type_names = {"Payload"};
    options.types_namespace = "test::upload";
    auto result = compile(unique_dir("upload_padding"), R"(
struct Inner { float3 normal; float weight; };
struct Payload { float head; Inner nested; float2 values[2]; float tail; };
StructuredBuffer<Payload> data;
[shader("fragment")]
float4 fs_main() : SV_Target0 {
    return float4(data[0].nested.normal * data[0].nested.weight, data[0].tail + data[0].head + data[0].values[1].x);
}
)",
                          options);
    INFO(result.diagnostics);
    REQUIRE(result.success);
    CHECK(contains(result.types_header, "offsetof(Inner, weight) == 12"));
    CHECK(contains(result.types_header, "offsetof(Payload, nested) == 16"));
    CHECK(contains(result.types_header, "offsetof(Payload, values) == 32"));
    CHECK(contains(result.types_header, "offsetof(Payload, tail) == 48"));
    CHECK(contains(result.types_header, "sizeof(Payload) == 64"));
    CHECK(contains(result.types_header, "std::byte _pts_padding_0[12]"));
}

TEST_CASE("upload generation rejects layouts that need a matrix conversion") {
    SlangCompileOptions options;
    options.type_names = {"Payload"};
    options.types_namespace = "test_upload";
    auto result = compile(unique_dir("upload_matrix"), R"(
struct Payload { float3x3 matrix; };
StructuredBuffer<Payload> data;
[shader("fragment")]
float4 fs_main() : SV_Target0 { return float4(data[0].matrix[0], 1); }
)",
                          options);
    CHECK_FALSE(result.success);
    CHECK(result.types_header.empty());
    CHECK(contains(result.diagnostics, "matrix requires a packed column-major"));
}

TEST_CASE("upload generation rejects a type missing from reflected buffers") {
    SlangCompileOptions options;
    options.type_names = {"typo"};
    options.types_namespace = "test_upload";
    auto result = compile(unique_dir("upload_missing"), R"(
[shader("fragment")] float4 fs_main() : SV_Target0 { return float4(1); }
)",
                          options);
    CHECK_FALSE(result.success);
    CHECK(contains(result.diagnostics, "type not found in reflected WGSL buffers: typo"));
}

TEST_CASE("upload types can select nested array elements without emitting unrelated types") {
    SlangCompileOptions options;
    options.type_names = {"Inner", "Inner"};
    options.types_namespace = "test_upload";
    auto result = compile(unique_dir("upload_nested"), R"(
struct Inner { float4 value; };
struct Payload { Inner values[2]; };
struct Unrelated { float4 other; };
ConstantBuffer<Payload> uniforms;
StructuredBuffer<Payload> storage;
ConstantBuffer<Unrelated> extra;
[shader("fragment")]
float4 fs_main() : SV_Target0 {
    return uniforms.values[0].value + storage[0].values[1].value + extra.other;
}
)",
                          options);
    INFO(result.diagnostics);
    REQUIRE(result.success);
    const std::string declaration = "struct alignas(16) Inner";
    const auto first = result.types_header.find(declaration);
    REQUIRE(first != std::string::npos);
    CHECK(result.types_header.find(declaration, first + 1) == std::string::npos);
    CHECK_FALSE(contains(result.types_header, "Payload"));
    CHECK_FALSE(contains(result.types_header, "Unrelated"));
}

TEST_CASE("upload type selection rejects a binding name") {
    SlangCompileOptions options;
    options.type_names = {"data"};
    options.types_namespace = "test_upload";
    auto result = compile(unique_dir("upload_binding_name"), R"(
struct Payload { float4 value; };
ConstantBuffer<Payload> data;
[shader("fragment")]
float4 fs_main() : SV_Target0 { return data.value; }
)",
                          options);
    CHECK_FALSE(result.success);
    CHECK(result.types_header.empty());
    CHECK(contains(result.diagnostics, "type not found in reflected WGSL buffers: data"));
}

TEST_CASE("upload types reject incompatible structs with the same reflected name") {
    SlangCompileOptions options;
    options.type_names = {"Payload"};
    options.types_namespace = "test_upload";
    std::string field_type;
    SUBCASE("different sizes") {
        field_type = "float2";
    }
    SUBCASE("same size and offsets with a different scalar type") {
        field_type = "uint4";
    }
    auto result = compile(unique_dir("upload_conflict"), std::string(R"(
namespace A { struct Payload { float4 value; }; }
namespace B { struct Payload { )") + field_type + R"( value; }; }
ConstantBuffer<A::Payload> first;
StructuredBuffer<B::Payload> second;
[shader("fragment")]
float4 fs_main() : SV_Target0 { return first.value + float(second[0].value.x); }
)",
                          options);
    INFO(result.diagnostics);
    CHECK_FALSE(result.success);
    CHECK(result.types_header.empty());
    CHECK(contains(result.diagnostics, "conflicting buffer layouts for Payload"));
}

TEST_CASE("upload dependencies must match their uses in unselected buffer types") {
    SlangCompileOptions options;
    options.type_names = {"Payload"};
    options.types_namespace = "test_upload";
    auto result = compile(unique_dir("upload_dependency_conflict"), R"(
namespace A { struct Inner { float4 value; }; }
namespace B { struct Inner { float2 value; }; }
struct Payload { A::Inner nested; };
struct Unselected { B::Inner nested; };
ConstantBuffer<Payload> first;
StructuredBuffer<Unselected> second;
[shader("fragment")]
float4 fs_main() : SV_Target0 {
    return first.nested.value + float4(second[0].nested.value, 0, 0);
}
)",
                          options);
    INFO(result.diagnostics);
    CHECK_FALSE(result.success);
    CHECK(result.types_header.empty());
    CHECK(contains(result.diagnostics, "conflicting buffer layouts for Inner"));
}

TEST_CASE("C++ compilation emits callable exports without a compute entry point") {
    SlangCompileOptions options;
    options.cpp = true;
    auto result = compile(unique_dir("cpu_exports"), R"(
export __extern_cpp float exported_square(float x) { return x * x; }
)",
                          options);
    INFO(result.diagnostics);
    REQUIRE(result.success);
    CHECK(result.wgsl.empty());
    CHECK(contains(result.cpp, "exported_square"));
    CHECK(contains(result.cpp_header, "float exported_square(float"));
    // The prelude is embedded; cross-compiles must not depend on host SDK paths.
    CHECK_FALSE(contains(result.cpp, "#include \"C:"));
    CHECK_FALSE(contains(result.cpp_header, "#include \"C:"));
    CHECK(result.dependencies.size() >= 1);
}

TEST_CASE("metadata header emits ConstantBuffer as Uniform without hasDynamicOffset by default") {
    auto dir = unique_dir("uniform");
    auto result = compile(dir, R"(
struct Uniforms { float4 color; };
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> u;

[shader("fragment")]
float4 fs_main() : SV_Target0 { return u.color; }
)");
    REQUIRE(result.success);
    REQUIRE_FALSE(result.metadata_header.empty());
    CHECK(contains(result.metadata_header, ".buffer.type = WGPUBufferBindingType_Uniform"));
    CHECK_FALSE(contains(result.metadata_header, "hasDynamicOffset"));
}

TEST_CASE("metadata header emits hasDynamicOffset=true when [DynamicBuffer] is applied") {
    auto dir = unique_dir("dynbuf");
    auto result = compile(dir, R"(
struct Uniforms { float4 color; };
[[vk::binding(0, 0)]] [DynamicBuffer] ConstantBuffer<Uniforms> u;

[shader("fragment")]
float4 fs_main() : SV_Target0 { return u.color; }
)");
    REQUIRE(result.success);
    REQUIRE_FALSE(result.metadata_header.empty());
    CHECK(contains(result.metadata_header, ".buffer.type = WGPUBufferBindingType_Uniform"));
    CHECK(contains(result.metadata_header, ".buffer.hasDynamicOffset = true"));
}

TEST_CASE("metadata header emits texture + sampler bindings from Texture2D and SamplerState") {
    auto dir = unique_dir("tex");
    auto result = compile(dir, R"(
[[vk::binding(0, 0)]] Texture2D<float4> albedo;
[[vk::binding(1, 0)]] SamplerState samp;

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target0 {
    return albedo.Sample(samp, uv);
}
)");
    REQUIRE(result.success);
    REQUIRE_FALSE(result.metadata_header.empty());
    CHECK(contains(result.metadata_header, ".texture.sampleType = WGPUTextureSampleType_Float"));
    CHECK(contains(result.metadata_header, ".texture.viewDimension = WGPUTextureViewDimension_2D"));
    CHECK(contains(result.metadata_header, ".sampler.type = WGPUSamplerBindingType_Filtering"));
    // Textures and samplers should NOT produce buffer.type entries.
    CHECK_FALSE(
        contains(result.metadata_header, "entry0.buffer.type = WGPUBufferBindingType_Uniform"));
}

TEST_CASE("metadata header emits ReadOnlyStorage for StructuredBuffer and Storage for RW") {
    auto dir = unique_dir("sbuf");
    auto result = compile(dir, R"(
struct Particle { float4 pos; };
[[vk::binding(0, 0)]] StructuredBuffer<Particle>   particles_ro;
[[vk::binding(1, 0)]] RWStructuredBuffer<Particle> particles_rw;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    particles_rw[tid.x] = particles_ro[tid.x];
}
)");
    REQUIRE(result.success);
    REQUIRE_FALSE(result.metadata_header.empty());
    CHECK(contains(result.metadata_header, ".buffer.type = WGPUBufferBindingType_ReadOnlyStorage"));
    CHECK(contains(result.metadata_header, ".buffer.type = WGPUBufferBindingType_Storage"));
}

TEST_CASE("metadata header emits storageTexture for RWTexture2D") {
    auto dir = unique_dir("rwtex");
    auto result = compile(dir, R"(
[[vk::binding(0, 0)]] RWTexture2D<float4> out_img;

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    out_img[tid.xy] = float4(1.0, 0.0, 0.0, 1.0);
}
)");
    REQUIRE(result.success);
    REQUIRE_FALSE(result.metadata_header.empty());
    CHECK(contains(result.metadata_header, ".storageTexture.access"));
    CHECK(contains(result.metadata_header,
                   ".storageTexture.viewDimension = WGPUTextureViewDimension_2D"));
}
