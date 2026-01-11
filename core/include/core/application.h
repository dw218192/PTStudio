#pragma once
#include <fmt/core.h>

#include <deque>
#include <sstream>

#include "logging.h"
#include "utils.h"

namespace PTS {

/**
 * \brief The base class for all applications (not necessarily graphical)
 */
struct Application {
    DEFAULT_COPY_MOVE(Application);

    Application(std::string_view name) noexcept : m_name{name.begin(), name.end()} {
    }
    virtual ~Application() = default;
    virtual void run() = 0;
    NODISCARD virtual auto get_name() const noexcept -> std::string_view {
        return m_name;
    }
    NODISCARD virtual auto get_time() const noexcept -> float {
        return 0.0f;
    }
    NODISCARD virtual auto get_delta_time() const noexcept -> float {
        return 0.0f;
    }
    /**
     * \brief Terminates the program with the given exit code
     * \param code the exit code
     */
    [[noreturn]] virtual void quit(int code) = 0;

    template <typename... Args>
    void log(Logging::LogLevel level, std::string_view fmt, Args&&... args) noexcept {
        Logging::get_logger(get_name())
            .log(static_cast<spdlog::level::level_enum>(level), fmt, std::forward<Args>(args)...);
    }

   private:
    std::string m_name;
};
}  // namespace PTS
