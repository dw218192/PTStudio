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
          m_logger(host_api && host_api->create_logger ? host_api->create_logger(logger_name)
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

#define PTS_PLUGIN_LOG_METHOD(method_name, level)                               \
    template <typename... Args>                                                 \
    void method_name(fmt::format_string<Args...> format, Args&&... args) const   \
        noexcept {                                                              \
        log_impl(level, format, std::forward<Args>(args)...);                   \
    }

    PTS_PLUGIN_LOG_METHOD(log_info, PTS_LOG_LEVEL_INFO)
    PTS_PLUGIN_LOG_METHOD(log_warning, PTS_LOG_LEVEL_WARNING)
    PTS_PLUGIN_LOG_METHOD(log_error, PTS_LOG_LEVEL_ERROR)
    PTS_PLUGIN_LOG_METHOD(log_critical, PTS_LOG_LEVEL_CRITICAL)
    PTS_PLUGIN_LOG_METHOD(log_debug, PTS_LOG_LEVEL_DEBUG)
    PTS_PLUGIN_LOG_METHOD(log_trace, PTS_LOG_LEVEL_TRACE)

#undef PTS_PLUGIN_LOG_METHOD

    [[nodiscard]] auto is_level_enabled(PtsLogLevel level) const noexcept -> bool {
        if (!m_logger || !m_host_api || !m_host_api->is_level_enabled) {
            return false;
        }
        return m_host_api->is_level_enabled(m_logger, level);
    }

   private:
    template <typename... Args>
    void log_impl(PtsLogLevel level, fmt::format_string<Args...> format,
                  Args&&... args) const noexcept {
        if (!m_logger || !m_host_api || !m_host_api->log) {
            return;
        }

        if (m_host_api->is_level_enabled) {
            if (!m_host_api->is_level_enabled(m_logger, level)) {
                return;
            }
        } else if (level == PTS_LOG_LEVEL_OFF) {
            return;
        }

        std::string message = fmt::format(format, std::forward<Args>(args)...);
        m_host_api->log(m_logger, level, message.c_str());
    }

    PtsHostApi* m_host_api = nullptr;
    LoggerHandle m_logger = nullptr;
};

[[nodiscard]] inline auto make_logger(PtsHostApi* host_api,
                                      const char* logger_name) noexcept -> PluginLogger {
    return PluginLogger(host_api, logger_name);
}

}  // namespace pts
