#pragma once
#include <core/defines.h>
#include <core/enumUtils.h>
#include <spdlog/spdlog.h>

#include <boost/describe/enum.hpp>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace pts {
enum class LogLevel {
    Trace = SPDLOG_LEVEL_TRACE,
    Debug = SPDLOG_LEVEL_DEBUG,
    Info = SPDLOG_LEVEL_INFO,
    Warning = SPDLOG_LEVEL_WARN,
    Error = SPDLOG_LEVEL_ERROR,
    Critical = SPDLOG_LEVEL_CRITICAL,
    Off = SPDLOG_LEVEL_OFF,
};

BOOST_DESCRIBE_ENUM(LogLevel, Trace, Debug, Info, Warning, Error, Critical, Off);

/**
 * @brief Configuration for the logging system
 */
struct Config {
    LogLevel level{LogLevel::Info};
    std::string pattern{"[%H:%M:%S] [%^%L%$] [%n] %v"};
};

class LoggingManager {
   public:
    LoggingManager(Config const& config);
    ~LoggingManager();

    NO_COPY_MOVE(LoggingManager);
    /**
     * @brief Gets a lazily initialized logger with the given name. The logger will be registered
     * and kept alive until the program exits.
     * @param name The name of the logger (default: "pts")
     * @return The logger
     */
    [[nodiscard]] auto get_logger(std::string_view name = "pts") noexcept -> spdlog::logger&;

    /**
     * @brief Gets a shared pointer to a lazily initialized logger with the given name.
     * The logger will be registered and kept alive until the program exits.
     * @param name The name of the logger
     * @return A shared pointer to the logger
     */
    [[nodiscard]] auto get_logger_shared(std::string_view name) noexcept
        -> std::shared_ptr<spdlog::logger>;

    /**
     * @brief Adds a sink to the logging system.
     * @param sink The sink to add.
     */
    void add_sink(spdlog::sink_ptr sink);

   private:
    std::mutex mtx_;
    Config cfg_;
    std::vector<spdlog::sink_ptr> sinks_;
};

/**
 * Helper function to log to a logger or stderr if the logger gets outlived by the calling code.
 * @param name The name of the logger
 * @param level The level of the log
 * @param fmt The format string
 * @param args The arguments to format the string
 */
template <typename... Args>
inline void log_or_cerr(const std::string& name, LogLevel level, std::string_view fmt,
                        Args&&... args) {
    if (auto lg = spdlog::get(name)) {
        auto spd_level = static_cast<spdlog::level::level_enum>(level);
        if (spdlog::should_log(spd_level)) {
            lg->log(spd_level, fmt, std::forward<Args>(args)...);
        }
    } else {
        std::cerr << '[' << pts::to_string(level) << "] "
                  << "[" << name << "] " << fmt::format(fmt, std::forward<Args>(args)...) << "\n";
    }
}
}  // namespace pts
