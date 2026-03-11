#pragma once
#include <core/loggingManager.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>

namespace pts {

class CommandLine;

/**
 * @brief Headless base class for all applications.
 *
 * Provides logging, an event loop (with Emscripten support), and
 * frame-rate timing.  Subclasses add windowing and rendering as
 * needed (see WindowedApplication).
 */
struct Application {
    NO_COPY_MOVE(Application);

    Application(std::string_view name, pts::LoggingManager& logging_manager);
    virtual ~Application();

    /**
     * @brief Initialize command-line arguments and process them.
     *
     * Creates a CommandLine, calls register_args() for virtual dispatch,
     * parses argc/argv, then calls process_args(). Returns false if --help
     * was requested (caller should exit).
     */
    [[nodiscard]] bool init(int argc, char* argv[]);

    /**
     * @brief Register command-line arguments. Override to add app-specific args.
     *
     * Base implementation registers --num-frames.
     * Derived classes should call the base version first.
     */
    virtual void register_args(CommandLine& cli);

    /**
     * @brief Process parsed command-line arguments. Override to read app-specific args.
     *
     * Base implementation reads --num-frames and calls set_max_frames().
     * Derived classes should call the base version first.
     */
    virtual void process_args(const CommandLine& cli);

    virtual void run();

    [[nodiscard]] auto get_name() const noexcept -> std::string_view {
        return m_name;
    }
    [[nodiscard]] auto get_time() const noexcept -> float;
    [[nodiscard]] auto get_delta_time() const noexcept -> float;

    void set_min_frame_time(float min_frame_time) noexcept;

    /// Quit after this many frames (0 = unlimited).
    void set_max_frames(int n) noexcept;

    /// Signal the event loop to stop after the current frame.
    void request_stop() noexcept;

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
    auto get_logger() noexcept -> std::shared_ptr<spdlog::logger> {
        return m_logger;
    }

    /// Returns true when request_stop() has been called.
    [[nodiscard]] bool should_stop() const noexcept;

    /// Increment the frame counter; calls request_stop() when the limit is hit.
    void check_frame_limit() noexcept;

    // Common args accessible to subclasses (set by process_args)
    unsigned m_width = 1280;
    unsigned m_height = 720;

    /**
     * @brief Process a single frame. Override in derived classes for custom frame behavior.
     *
     * Base implementation: compute timing, call loop(dt), enforce frame-rate cap.
     */
    virtual void run_one_frame();

    float m_min_frame_time{0.0f};

   private:
    // Class invariants:
    // - m_logging_manager is always valid (non-null)
    // - m_logger is always valid (non-null)

    std::string m_name;
    pts::LoggingManager* m_logging_manager;
    std::shared_ptr<spdlog::logger> m_logger;

    std::chrono::steady_clock::time_point m_start_time;
    float m_delta_time{0.0f};
    int m_max_frames{0};
    int m_frame_count{0};
    bool m_should_stop{false};
};
}  // namespace pts
