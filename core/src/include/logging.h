#pragma once
#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace PTS {
/**
 * \brief Configuration for the logging system
 */
struct LoggingConfig {
    spdlog::level::level_enum level{spdlog::level::info};
    std::string pattern{"[%H:%M:%S] [%^%L%$] %v"};
};

/**
 * \brief Gets a lazily initialized logger with the given name
 * \param name The name of the logger
 * \return The logger
 */
[[nodiscard]] auto get_logger(std::string_view name) noexcept -> spdlog::logger&;

/**
 * \brief Initializes the logging system. This should be called before any logging is done.
 * \note This should be called only once, but this method is idempotent.
 */
void init_logging(LoggingConfig const& config);
}  // namespace PTS
