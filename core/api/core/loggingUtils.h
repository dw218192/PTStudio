#pragma once

#include <core/plugin.h>
#include <fmt/format.h>

#include <string>
#include <utility>

namespace pts {

/**
 * C++ convenience wrapper over the C ABI logging functions in PtsHostApi.
 */
class PluginLogger final {
   public:
    PluginLogger() = default;

    PluginLogger(PtsHostApi* host_api, const char* logger_name) noexcept
        : m_host_api(host_api),
          m_logger(host_api && host_api->create_logger
                       ? host_api->create_logger(logger_name)
                       : nullptr) {
    }

    PluginLogger(PtsHostApi* host_api, LoggerHandle logger) noexcept
        : m_host_api(host_api), m_logger(logger) {
    }

    [[nodiscard]] auto is_valid() const noexcept -> bool {
        return m_host_api && m_logger;
    }

    [[nodiscard]] auto handle() const noexcept -> LoggerHandle {
        return m_logger;
    }

    template <typename... Args>
    void log_info(fmt::format_string<Args...> format, Args&&... args) const noexcept {
        log_impl(PTS_LOG_LEVEL_INFO, m_host_api ? m_host_api->log_info : nullptr, format,
                 std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log_warning(fmt::format_string<Args...> format, Args&&... args) const noexcept {
        log_impl(PTS_LOG_LEVEL_WARNING, m_host_api ? m_host_api->log_warning : nullptr, format,
                 std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log_error(fmt::format_string<Args...> format, Args&&... args) const noexcept {
        log_impl(PTS_LOG_LEVEL_ERROR, m_host_api ? m_host_api->log_error : nullptr, format,
                 std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log_critical(fmt::format_string<Args...> format, Args&&... args) const noexcept {
        log_impl(PTS_LOG_LEVEL_CRITICAL, m_host_api ? m_host_api->log_critical : nullptr, format,
                 std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log_debug(fmt::format_string<Args...> format, Args&&... args) const noexcept {
        log_impl(PTS_LOG_LEVEL_DEBUG, m_host_api ? m_host_api->log_debug : nullptr, format,
                 std::forward<Args>(args)...);
    }

    template <typename... Args>
    void log_trace(fmt::format_string<Args...> format, Args&&... args) const noexcept {
        log_impl(PTS_LOG_LEVEL_TRACE, m_host_api ? m_host_api->log_trace : nullptr, format,
                 std::forward<Args>(args)...);
    }

    [[nodiscard]] auto is_level_enabled(PtsLogLevel level) const noexcept -> bool {
        if (!m_logger || !m_host_api || !m_host_api->is_level_enabled) {
            return false;
        }
        return m_host_api->is_level_enabled(m_logger, level);
    }

   private:
    template <typename... Args>
    void log_impl(PtsLogLevel level, void (*log_fn)(LoggerHandle, const char*),
                  fmt::format_string<Args...> format, Args&&... args) const noexcept {
        if (!m_logger || !log_fn) {
            return;
        }

        if (m_host_api && m_host_api->is_level_enabled) {
            if (!m_host_api->is_level_enabled(m_logger, level)) {
                return;
            }
        } else if (level == PTS_LOG_LEVEL_OFF) {
            return;
        }
        
        std::string message = fmt::format(format, std::forward<Args>(args)...);
        log_fn(m_logger, message.c_str());
    }

    PtsHostApi* m_host_api = nullptr;
    LoggerHandle m_logger = nullptr;
};

[[nodiscard]] inline auto make_logger(PtsHostApi* host_api, const char* logger_name) noexcept
    -> PluginLogger {
    return PluginLogger(host_api, logger_name);
}

}  // namespace pts
