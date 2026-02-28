#include <core/application.h>
#include <core/commandLine.h>
#include <core/diagnostics.h>
#include <core/timeUtils.h>

#include <chrono>
#include <thread>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

namespace pts {

Application::Application(std::string_view name, pts::LoggingManager& logging_manager,
                         pts::PluginManager& plugin_manager, float min_frame_time)
    : m_name{name.begin(), name.end()},
      m_logging_manager{&logging_manager},
      m_plugin_manager{&plugin_manager} {
    m_logger = m_logging_manager->get_logger_shared(get_name().data());
    INVARIANT_MSG(m_logger != nullptr, "get_logger_shared must return valid logger");
    set_min_frame_time(min_frame_time);
    m_start_time = std::chrono::steady_clock::now();
}

bool Application::init(int argc, char* argv[]) {
    CommandLine cli;
    register_args(cli);
    if (!cli.parse(argc, argv)) {
        return false;
    }
    process_args(cli);
    return true;
}

void Application::register_args(CommandLine& cli) {
    cli.add_int("num-frames", "Quit after N frames (0 = unlimited)", 0);
}

void Application::process_args(const CommandLine& cli) {
    set_max_frames(cli.get_int("num-frames"));
}

Application::~Application() = default;

void Application::run() {
#if defined(__EMSCRIPTEN__)
    // Emscripten requires yielding control to the browser via main loop callback
    emscripten_set_main_loop_arg(
        [](void* arg) {
            auto* app = static_cast<Application*>(arg);
            app->run_one_frame();
            app->check_frame_limit();
        },
        this, 0, true);
#else
    while (!m_should_stop) {
        run_one_frame();
        check_frame_limit();
    }
#endif
}

void Application::run_one_frame() {
    auto const frame_start = std::chrono::steady_clock::now();

    loop(m_delta_time);

    auto const frame_end = std::chrono::steady_clock::now();
    auto const frame_duration = std::chrono::duration<float>(frame_end - frame_start).count();
    m_delta_time = frame_duration;

#if !defined(__EMSCRIPTEN__)
    if (m_min_frame_time > 0.0f && frame_duration < m_min_frame_time) {
        auto const sleep_duration = m_min_frame_time - frame_duration;
        std::this_thread::sleep_for(
            std::chrono::duration<float, std::milli>(sleep_duration * 1000.0f));
        m_delta_time = m_min_frame_time;
    }
#endif
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

void Application::set_max_frames(int n) noexcept {
    m_max_frames = n;
}

auto Application::should_stop() const noexcept -> bool {
    return m_should_stop;
}

void Application::check_frame_limit() noexcept {
    if (m_max_frames > 0 && ++m_frame_count >= m_max_frames) {
        request_stop();
    }
}

void Application::request_stop() noexcept {
    m_should_stop = true;
#if defined(__EMSCRIPTEN__)
    emscripten_cancel_main_loop();
#endif
}

}  // namespace pts
