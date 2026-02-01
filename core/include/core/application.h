#pragma once
#include <core/loggingManager.h>
#include <core/rendering/windowing.h>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <deque>
#include <memory>
#include <sstream>

#include "pluginManager.h"

namespace pts {
namespace rendering {
class WebGpuContext;
class IWindowing;
class IViewport;
}  // namespace rendering

/**
 * @brief Base class for all applications with WebGPU rendering and windowing.
 */
struct Application {
    NO_COPY_MOVE(Application);

    Application(std::string_view name, pts::LoggingManager& logging_manager,
                pts::PluginManager& plugin_manager, unsigned width, unsigned height,
                float min_frame_time);
    virtual ~Application();

    virtual void run();

    [[nodiscard]] auto get_name() const noexcept -> std::string_view {
        return m_name;
    }
    [[nodiscard]] auto get_window_width() const noexcept -> int;
    [[nodiscard]] auto get_window_height() const noexcept -> int;
    [[nodiscard]] auto get_time() const noexcept -> float;
    [[nodiscard]] auto get_delta_time() const noexcept -> float;

    void set_min_frame_time(float min_frame_time) noexcept;
    void on_framebuffer_resized() noexcept;

    /**
     * @brief Called every frame. Override to handle the main loop.
     * @param dt the time since the last frame
     */
    virtual void loop(float dt) = 0;

    template <typename... Args>
    void log(pts::LogLevel level, std::string_view fmt, Args&&... args) noexcept {
        m_logger->log(static_cast<spdlog::level::level_enum>(level), fmt,
                      std::forward<Args>(args)...);
    }

   protected:
    pts::LoggingManager& get_logging_manager() noexcept {
        return *m_logging_manager;
    }
    pts::PluginManager& get_plugin_manager() noexcept {
        return *m_plugin_manager;
    }
    auto get_logger() noexcept -> std::shared_ptr<spdlog::logger> {
        return m_logger;
    }

    [[nodiscard]] auto get_webgpu_context() noexcept -> pts::rendering::WebGpuContext* {
        return m_webgpu_context.get();
    }
    [[nodiscard]] auto get_webgpu_context() const noexcept -> const pts::rendering::WebGpuContext* {
        return m_webgpu_context.get();
    }

    [[nodiscard]] auto get_windowing() noexcept -> pts::rendering::IWindowing* {
        return m_windowing.get();
    }
    [[nodiscard]] auto get_windowing() const noexcept -> const pts::rendering::IWindowing* {
        return m_windowing.get();
    }

    [[nodiscard]] auto get_viewport() noexcept -> pts::rendering::IViewport* {
        return m_viewport.get();
    }
    [[nodiscard]] auto get_viewport() const noexcept -> const pts::rendering::IViewport* {
        return m_viewport.get();
    }

    void set_framebuffer_resized(bool value) noexcept {
        m_framebuffer_resized = value;
    }

   private:
    // Class invariants (enforced in constructor, throw on failure):
    // - m_webgpu_context is always valid (non-null)
    // - m_windowing is always valid (non-null)
    // - m_viewport is always valid (non-null)
    // - m_logging_manager and m_plugin_manager are always valid (non-null)
    // - m_logger is always valid (non-null)

    std::string m_name;
    pts::LoggingManager* m_logging_manager;
    pts::PluginManager* m_plugin_manager;
    std::shared_ptr<spdlog::logger> m_logger;

    std::unique_ptr<pts::rendering::IWindowing> m_windowing;
    std::unique_ptr<pts::rendering::IViewport> m_viewport;
    std::unique_ptr<pts::rendering::WebGpuContext> m_webgpu_context;
    bool m_framebuffer_resized{false};
    std::chrono::steady_clock::time_point m_start_time;
    float m_min_frame_time{0.0f};
    float m_delta_time{0.0f};
};
}  // namespace pts
