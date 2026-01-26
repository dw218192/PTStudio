#pragma once

#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/texture.h>

#include <string>
#include <string_view>

namespace pts::webgpu {

class Device {
   public:
    Device() = default;
    explicit Device(WGPUInstance instance, WGPUDevice device, WGPUQueue queue);

    Device(const Device&) = delete;
    auto operator=(const Device&) -> Device& = delete;

    Device(Device&& other) noexcept;
    auto operator=(Device&& other) noexcept -> Device&;

    ~Device();

    [[nodiscard]] static auto create() -> Device;
    [[nodiscard]] auto is_valid() const noexcept -> bool;
    [[nodiscard]] auto handle() const noexcept -> WGPUDevice;
    [[nodiscard]] auto queue() const noexcept -> WGPUQueue;

    [[nodiscard]] auto create_buffer(std::size_t size, WGPUBufferUsage usage) const -> Buffer;
    [[nodiscard]] auto create_shader_module(std::string_view wgsl_path) const -> ShaderModule;

   private:
    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;
    WGPUInstance m_instance = nullptr;
};

}  // namespace pts::webgpu
