#include <core/application.h>
#include <core/commandLine.h>
#include <core/diagnostics.h>
#include <core/profiling.h>
#include <core/timeUtils.h>

#include <chrono>
#include <thread>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

namespace pts {

Application::Application(std::string_view name, pts::LoggingManager& logging_manager)
    : m_name{name.begin(), name.end()}, m_logging_manager{&logging_manager} {
    PTS_STARTUP_PROFILER();
    m_logger = m_logging_manager->get_logger_shared(get_name().data());
    INVARIANT_MSG(m_logger != nullptr, "get_logger_shared must return valid logger");
    m_start_time = std::chrono::steady_clock::now();
}

auto Application::init(int argc, char* argv[]) -> std::optional<int> {
    CommandLine cli;
    register_args(cli);
    if (!cli.parse(argc, argv)) {
        return 0;  // --help was shown
    }
    auto ec = process_args(cli);
    if (ec != ErrorCode::Ok) {
        return static_cast<int>(ec);
    }
    return std::nullopt;
}

void Application::register_args(CommandLine& cli) {
    cli.add_int("num-frames", "Quit after N frames (0 = unlimited)", 0);
    cli.add_int("width", "Window width", 1280);
    cli.add_int("height", "Window height", 720);
    cli.add_int("max-fps", "Maximum frames per second (0 = unlimited)", 0);
    // --log-level is handled in main() before Application construction
    // (LoggingManager must exist first). Registered here so --help shows it.
    cli.add_string("log-level", "Log level (trace, debug, info, warn, error, critical)");
}

auto Application::process_args(const CommandLine& cli) -> ErrorCode {
    set_max_frames(cli.get_int("num-frames"));
    auto width = cli.get_int("width", 1280);
    auto height = cli.get_int("height", 720);
    m_width = static_cast<unsigned>(width > 0 ? width : 1280);
    m_height = static_cast<unsigned>(height > 0 ? height : 720);
    auto max_fps = cli.get_int("max-fps", 0);
    if (max_fps > 0) {
        set_min_frame_time(1.0f / static_cast<float>(max_fps));
    }
    return ErrorCode::Ok;
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
