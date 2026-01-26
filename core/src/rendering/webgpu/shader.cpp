#include <core/rendering/webgpu/shader.h>

namespace pts::webgpu {

ShaderModule::ShaderModule(WGPUShaderModule shader_module) : m_shader_module(shader_module) {
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

auto ShaderModule::is_valid() const noexcept -> bool {
    return m_shader_module != nullptr;
}

auto ShaderModule::handle() const noexcept -> WGPUShaderModule {
    return m_shader_module;
}

}  // namespace pts::webgpu
