#pragma once

#include <core/rendering/webgpu/webgpu.h>

namespace pts::webgpu {

/// RAII wrapper for WGPUShaderModule.
/// Invariant: handle is non-null after construction (moved-from state is null but unusable).
class ShaderModule {
   public:
    explicit ShaderModule(WGPUShaderModule shader_module);

    ShaderModule(const ShaderModule&) = delete;
    auto operator=(const ShaderModule&) -> ShaderModule& = delete;

    ShaderModule(ShaderModule&& other) noexcept;
    auto operator=(ShaderModule&& other) noexcept -> ShaderModule&;

    ~ShaderModule();

    [[nodiscard]] auto handle() const noexcept -> WGPUShaderModule;

   private:
    WGPUShaderModule m_shader_module;
};

}  // namespace pts::webgpu
