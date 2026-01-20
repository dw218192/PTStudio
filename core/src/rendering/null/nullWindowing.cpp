#include <core/loggingManager.h>
#include <core/rendering/windowing.h>

#include <memory>
#include <string>

namespace pts::rendering {
namespace {
class NullViewport final : public IViewport {
   public:
    explicit NullViewport(const ViewportDesc& desc)
        : m_title(desc.title ? desc.title : ""),
          m_drawable{desc.width, desc.height},
          m_logical{desc.width, desc.height},
          m_visible(desc.visible) {
    }

    [[nodiscard]] auto native_handle() const noexcept -> NativeViewportHandle override {
        auto handle = NativeViewportHandle{};
        handle.windowing = WindowingType::null_backend;
        handle.platform = NativePlatform::emscripten;
        handle.windowing_handle = nullptr;
        handle.web.canvas_selector = nullptr;
        return handle;
    }

    [[nodiscard]] auto drawable_extent() const noexcept -> Extent2D override {
        return m_drawable;
    }

    [[nodiscard]] auto logical_extent() const noexcept -> Extent2D override {
        return m_logical;
    }

    [[nodiscard]] auto content_scale() const noexcept -> float override {
        return 1.0f;
    }

    [[nodiscard]] auto should_close() const noexcept -> bool override {
        return m_should_close;
    }

    void request_close() noexcept override {
        m_should_close = true;
        on_close_requested();
    }

    void set_title(const char* utf8) override {
        if (utf8) {
            m_title = utf8;
        }
    }

    void set_visible(bool v) override {
        m_visible = v;
    }

    void set_cursor_pos(double, double) noexcept override {
    }

   private:
    std::string m_title;
    Extent2D m_drawable{};
    Extent2D m_logical{};
    bool m_visible{true};
    bool m_should_close{false};
};

class NullWindowing final : public IWindowing {
   public:
    explicit NullWindowing(pts::LoggingManager&) {
    }

    [[nodiscard]] std::unique_ptr<IViewport> create_viewport(const ViewportDesc& desc) override {
        auto viewport = std::make_unique<NullViewport>(desc);
        m_primary = viewport.get();
        return viewport;
    }

    [[nodiscard]] NativeViewportHandle native_handle() const noexcept override {
        if (!m_primary) {
            auto handle = NativeViewportHandle{};
            handle.windowing = WindowingType::null_backend;
            handle.platform = NativePlatform::emscripten;
            return handle;
        }
        return m_primary->native_handle();
    }

    void pump_events(PumpEventMode) override {
    }

    [[nodiscard]] WindowingVulkanExtensions required_vulkan_instance_extensions()
        const noexcept override {
        return {};
    }

   private:
    IViewport* m_primary{nullptr};
};
}  // namespace

auto create_windowing(pts::LoggingManager& logging_manager) -> std::unique_ptr<IWindowing> {
    return std::make_unique<NullWindowing>(logging_manager);
}
}  // namespace pts::rendering
