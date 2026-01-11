#pragma once
#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

#include "utils.h"

namespace PTS {
namespace Logging {
enum class LogLevel {
    Trace = SPDLOG_LEVEL_TRACE,
    Debug = SPDLOG_LEVEL_DEBUG,
    Info = SPDLOG_LEVEL_INFO,
    Warning = SPDLOG_LEVEL_WARN,
    Error = SPDLOG_LEVEL_ERROR,
    Critical = SPDLOG_LEVEL_CRITICAL,
    Off = SPDLOG_LEVEL_OFF,
};

/**
 * \brief Configuration for the logging system
 */
struct Config {
    LogLevel level{Logging::LogLevel::Info};
    std::string pattern{"[%H:%M:%S] [%^%L%$] %v"};
};

/**
 * \brief Gets a lazily initialized logger with the given name
 * \param name The name of the logger
 * \return The logger
 */
NODISCARD auto get_logger(std::string_view name) noexcept -> spdlog::logger&;

/**
 * \brief Initializes the logging system. This should be called before any logging is done.
 * \note This should be called only once, but this method is idempotent.
 */
void init_logging(Config const& config);

/**
 * \brief Adds a sink to the logging system.
 * \param sink The sink to add.
 */
void add_sink(spdlog::sink_ptr sink);
}  // namespace Logging
}  // namespace PTS
