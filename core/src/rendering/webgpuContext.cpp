#include <core/loggingManager.h>
#include <core/rendering/webgpuContext.h>

namespace pts::rendering {

WebGpuContext::WebGpuContext(pts::webgpu::Device device, pts::webgpu::Surface surface)
    : m_device(std::move(device)), m_surface(std::move(surface)) {
}

WebGpuContext::~WebGpuContext() = default;

WebGpuContext::WebGpuContext(WebGpuContext&&) noexcept = default;
auto WebGpuContext::operator=(WebGpuContext&&) noexcept -> WebGpuContext& = default;

auto WebGpuContext::create(const IViewport& viewport,
                           pts::LoggingManager& logging_manager) -> std::unique_ptr<WebGpuContext> {
    auto logger = logging_manager.get_logger_shared("webgpu");

    try {
        // Device creation enforces invariants and throws on failure
        auto device = pts::webgpu::Device::create(logger);

        auto extent = viewport.drawable_extent();
        auto surface = pts::webgpu::Surface::create(device, viewport.native_handle(), extent);
        if (!surface.is_valid()) {
            logger->error("Failed to create WebGPU surface");
            return nullptr;
        }

        logger->info("WebGPU context created successfully");
        return std::make_unique<WebGpuContext>(std::move(device), std::move(surface));
    } catch (const std::runtime_error& e) {
        logger->error("Failed to create WebGPU context: {}", e.what());
        return nullptr;
    }
}

auto WebGpuContext::is_valid() const noexcept -> bool {
    // Device is always valid if constructed successfully (enforced invariants)
    return m_surface.is_valid();
}

auto WebGpuContext::device() const noexcept -> const pts::webgpu::Device& {
    return m_device;
}

auto WebGpuContext::surface() noexcept -> pts::webgpu::Surface& {
    return m_surface;
}

auto WebGpuContext::surface_format() const noexcept -> WGPUTextureFormat {
    return m_surface.format();
}
}  // namespace pts::rendering
