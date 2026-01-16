#pragma once
#include <core/loggingManager.h>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <deque>
#include <sstream>

#include "pluginManager.h"
#include "utils.h"

namespace PTS {

/**
 * @brief The base class for all applications (not necessarily graphical)
 */
struct Application {
    DEFAULT_COPY_MOVE(Application);

    Application(std::string_view name, pts::LoggingManager& logging_manager,
                pts::PluginManager& plugin_manager) noexcept
        : m_name{name.begin(), name.end()},
          m_logging_manager{&logging_manager},
          m_plugin_manager{&plugin_manager} {
        m_logger = m_logging_manager->get_logger_shared(get_name().data());
    }
    virtual ~Application() = default;
    virtual void run() = 0;
    [[nodiscard]] virtual auto get_name() const noexcept -> std::string_view {
        return m_name;
    }
    [[nodiscard]] virtual auto get_time() const noexcept -> float {
        return 0.0f;
    }
    [[nodiscard]] virtual auto get_delta_time() const noexcept -> float {
        return 0.0f;
    }
    /**
     * @brief Terminates the program with the given exit code
     * @param code the exit code
     */
    [[noreturn]] virtual void quit(int code) = 0;

    template <typename... Args>
    void log(pts::LogLevel level, std::string_view fmt, Args&&... args) noexcept {
        // Note: This requires a logger to be registered with the name from get_name()
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

   private:
    std::string m_name;
    pts::LoggingManager* m_logging_manager;
    pts::PluginManager* m_plugin_manager;
    std::shared_ptr<spdlog::logger> m_logger;
};
}  // namespace PTS
