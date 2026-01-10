#include "include/logging.h"

#include <spdlog/sinks/stdout_color_sinks.h>

#include <atomic>

namespace PTS {
namespace {
std::atomic<bool> g_initialized{false};
LoggingConfig g_config{};
}  // namespace

auto get_logger(std::string_view name) noexcept -> spdlog::logger& {
    std::string logger_name{name};

    // Try to get existing logger
    auto existing_logger = spdlog::get(logger_name);
    if (existing_logger) {
        return *existing_logger;
    }

    // Create new logger if it doesn't exist
    // Use stdout_color_mt for thread-safe colored console logging
    // If another thread created it concurrently, this will throw, so catch and retry
    try {
        auto new_logger = spdlog::stdout_color_mt(logger_name);
        new_logger->set_level(g_config.level);
        new_logger->set_pattern(g_config.pattern);
        return *new_logger;
    } catch (spdlog::spdlog_ex& ex) {
        // maybe another thread created it concurrently
        existing_logger = spdlog::get(logger_name);
        if (existing_logger) {
            return *existing_logger;
        }

        // fallback to default logger
        fmt::print(stderr, "Failed to create logger, falling back to default logger: {}",
                   ex.what());
        auto default_logger = spdlog::default_logger();
        default_logger->set_level(g_config.level);
        default_logger->set_pattern(g_config.pattern);
        return *default_logger;
    }
}

void init_logging(LoggingConfig const& config) {
    if (g_initialized.exchange(true)) {
        return;
    }

    g_config = config;
}
}  // namespace PTS