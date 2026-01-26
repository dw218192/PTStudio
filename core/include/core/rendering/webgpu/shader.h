#pragma once

#include <core/rendering/webgpu/webgpu.h>

namespace pts::webgpu {

class ShaderModule {
   public:
    ShaderModule() = default;
    explicit ShaderModule(WGPUShaderModule shader_module);

    ShaderModule(const ShaderModule&) = delete;
    auto operator=(const ShaderModule&) -> ShaderModule& = delete;

    ShaderModule(ShaderModule&& other) noexcept;
    auto operator=(ShaderModule&& other) noexcept -> ShaderModule&;

    ~ShaderModule();

    [[nodiscard]] auto is_valid() const noexcept -> bool;
    [[nodiscard]] auto handle() const noexcept -> WGPUShaderModule;

   private:
    WGPUShaderModule m_shader_module = nullptr;
};

}  // namespace pts::webgpu
