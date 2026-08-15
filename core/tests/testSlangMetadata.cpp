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

SlangCompileOutput compile(const fs::path& dir, const std::string& slang_src) {
    auto path = write_slang(dir, "shader.slang", slang_src);
    Slang::ComPtr<slang::IGlobalSession> gs;
    REQUIRE(SLANG_SUCCEEDED(slang::createGlobalSession(gs.writeRef())));
    REQUIRE(gs);

    std::vector<std::string> entries;  // auto-discover
    std::vector<std::string_view> defines;
    std::vector<std::filesystem::path> search_paths{dir};
    return run_slang(gs.get(), search_paths, path, entries,
                     boost::span<const std::string_view>(defines.data(), defines.size()),
                     "test_shader");
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

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
