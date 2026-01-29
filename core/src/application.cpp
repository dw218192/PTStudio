#include <core/application.h>
#include <core/rendering/webgpuContext.h>
#include <core/rendering/windowing.h>

#include <chrono>
#include <thread>

namespace pts {
namespace {
auto time_since_start(const std::chrono::steady_clock::time_point& start) -> double {
    auto const now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}
}  // namespace

Application::Application(std::string_view name, pts::LoggingManager& logging_manager,
                         pts::PluginManager& plugin_manager, unsigned width, unsigned height,
                         float min_frame_time)
    : m_name{name.begin(), name.end()},
      m_logging_manager{&logging_manager},
      m_plugin_manager{&plugin_manager} {
    m_logger = m_logging_manager->get_logger_shared(get_name().data());
    set_min_frame_time(min_frame_time);
    m_start_time = std::chrono::steady_clock::now();

    // Create windowing system
    m_windowing = pts::rendering::create_windowing(get_logging_manager());

    // Create viewport
    auto viewport_desc = pts::rendering::ViewportDesc{
        get_name().data(), width, height, true, true, true, true,
    };
    m_viewport = m_windowing->create_viewport(viewport_desc);
    m_viewport->on_drawable_resized.connect(
        [this](pts::rendering::Extent2D) { on_framebuffer_resized(); });

    // Create WebGPU context
    m_webgpu_context = pts::rendering::WebGpuContext::create(*m_viewport, get_logging_manager());
    if (!m_webgpu_context || !m_webgpu_context->is_valid()) {
        throw std::runtime_error("Failed to create WebGPU context");
    }

    log(pts::LogLevel::Info, "Application initialized");
}

Application::~Application() = default;

void Application::run() {
    while (!m_viewport->should_close()) {
        auto const frame_start = std::chrono::steady_clock::now();

        m_windowing->pump_events(pts::rendering::PumpEventMode::Poll);

        loop(m_delta_time);

        if (m_framebuffer_resized) {
            auto const extent = m_viewport->drawable_extent();
            m_webgpu_context->surface().resize(extent);
            m_framebuffer_resized = false;
        }

        auto const frame_end = std::chrono::steady_clock::now();
        auto const frame_duration = std::chrono::duration<float>(frame_end - frame_start).count();
        m_delta_time = frame_duration;

        if (m_min_frame_time > 0.0f && frame_duration < m_min_frame_time) {
            auto const sleep_duration = m_min_frame_time - frame_duration;
            std::this_thread::sleep_for(
                std::chrono::duration<float, std::milli>(sleep_duration * 1000.0f));
            m_delta_time = m_min_frame_time;
        }
    }
}

auto Application::get_window_width() const noexcept -> int {
    return static_cast<int>(m_viewport->drawable_extent().w);
}

auto Application::get_window_height() const noexcept -> int {
    return static_cast<int>(m_viewport->drawable_extent().h);
}

auto Application::get_time() const noexcept -> float {
    return static_cast<float>(time_since_start(m_start_time));
}

auto Application::get_delta_time() const noexcept -> float {
    return m_delta_time;
}

void Application::set_min_frame_time(float min_frame_time) noexcept {
    m_min_frame_time = min_frame_time;
}

void Application::on_framebuffer_resized() noexcept {
    m_framebuffer_resized = true;
}

}  // namespace pts
