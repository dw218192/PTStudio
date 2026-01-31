#include <core/diagnostics.h>
#include <core/rendering/webgpu/shader.h>

namespace pts::webgpu {

ShaderModule::ShaderModule(WGPUShaderModule shader_module) : m_shader_module(shader_module) {
    INVARIANT_MSG(m_shader_module != nullptr, "handle is null");
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept : m_shader_module(other.m_shader_module) {
    other.m_shader_module = nullptr;
}

auto ShaderModule::operator=(ShaderModule&& other) noexcept -> ShaderModule& {
    if (this != &other) {
        if (m_shader_module != nullptr) {
            wgpuShaderModuleRelease(m_shader_module);
        }
        m_shader_module = other.m_shader_module;
        other.m_shader_module = nullptr;
    }
    return *this;
}

ShaderModule::~ShaderModule() {
    if (m_shader_module != nullptr) {
        wgpuShaderModuleRelease(m_shader_module);
    }
}

auto ShaderModule::handle() const noexcept -> WGPUShaderModule {
    ASSERT_MSG(m_shader_module != nullptr, "use after move");
    return m_shader_module;
}

}  // namespace pts::webgpu
