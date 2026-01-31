#pragma once

#include <core/rendering/webgpu/buffer.h>
#include <core/rendering/webgpu/pipeline.h>
#include <core/rendering/webgpu/shader.h>
#include <core/rendering/webgpu/texture.h>

#include <memory>
#include <string>
#include <string_view>

namespace spdlog {
class logger;
}

namespace pts::webgpu {

class Device {
   public:
    // Constructor enforces invariants: all handles must be non-null or throws std::runtime_error
    explicit Device(WGPUInstance instance, WGPUDevice device, WGPUQueue queue,
                    std::shared_ptr<spdlog::logger> logger);

    Device(const Device&) = delete;
    auto operator=(const Device&) -> Device& = delete;

    Device(Device&& other) noexcept;
    auto operator=(Device&& other) noexcept -> Device&;

    ~Device();

    // Factory method creates device with error callbacks registered, throws on failure
    [[nodiscard]] static auto create(std::shared_ptr<spdlog::logger> logger) -> Device;

    [[nodiscard]] auto instance() const noexcept -> WGPUInstance;
    [[nodiscard]] auto handle() const noexcept -> WGPUDevice;
    [[nodiscard]] auto queue() const noexcept -> WGPUQueue;

    [[nodiscard]] auto create_buffer(std::size_t size, WGPUBufferUsage usage) const -> Buffer;
    [[nodiscard]] auto create_shader_module(std::string_view wgsl_path) const -> ShaderModule;
    [[nodiscard]] auto create_shader_module_from_source(std::string_view wgsl_source) const
        -> ShaderModule;
    [[nodiscard]] auto create_pipeline_layout() const -> PipelineLayout;

   private:
    // invariants:
    // - m_instance is never null
    // - m_device is never null
    // - m_queue is never null
    // - m_logger is never null
    // - device lost and uncaptured error callbacks are registered

    WGPUInstance m_instance;
    WGPUDevice m_device;
    WGPUQueue m_queue;
    std::shared_ptr<spdlog::logger> m_logger;
};

}  // namespace pts::webgpu
