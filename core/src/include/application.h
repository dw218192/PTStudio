#pragma once
#include <deque>
#include <sstream>
#include <fmt/core.h>
#include "utils.h"

namespace PTS {
	DECL_ENUM(LogLevel,
	          Debug,
	          Info,
	          Warning,
	          Error,
	          StdOut,
	          StdErr
	);

	/**
	 * \brief The base class for all applications (not necessarily graphical)
	 * \details Provides an interface for logging and quitting the application
	*/
	struct Application {
		DEFAULT_COPY_MOVE(Application);

		Application() = default;
		virtual ~Application() = default;
		virtual void run() = 0;

		auto set_max_log_cnt(unsigned cnt) { m_max_line_cnt = cnt; }
		/**
		 * \brief Terminates the program with the given exit code
		 * \param code the exit code
		*/
		[[noreturn]] virtual void quit(int code) = 0;

		template <typename... Args>
		void log(LogLevel level, std::string_view fmt, Args&&... args) noexcept {
#ifdef NDEBUG
            if (level == LogLevel::Debug) {
                return;
            }
#endif
			try {
				if (level == LogLevel::StdOut) {
					fmt::println(fmt, std::forward<Args>(args)...);
					return;
				} else if (level == LogLevel::StdErr) {
					fmt::println(stderr, fmt, std::forward<Args>(args)...);
				}

				++m_line_cnt;

				if (m_line_cnt > m_max_line_cnt) {
					if (!m_logs.empty()) {
						m_logs.pop_front();
						--m_line_cnt;
					}
				}

				m_logs.emplace_back(level, fmt::format(fmt, std::forward<Args>(args)...));
			} catch (fmt::format_error const&) {
				m_logs.emplace_back(level, std::string{fmt});
			}
			on_log_added();
		}

		// optional methods
		NODISCARD virtual auto get_time() const noexcept -> float { return 0.0f; }
		NODISCARD virtual auto get_delta_time() const noexcept -> float { return 0.0f; }

	protected:
		struct LogDesc {
			LogLevel level;
			std::string msg;

			LogDesc(LogLevel level, std::string msg)
				: level{level}, msg{std::move(msg)} {}
		};

		virtual void on_log_added() = 0;
		NODISCARD auto get_logs() const -> View<std::deque<LogDesc>> { return m_logs; }
		auto clear_logs() { m_logs.clear(); }

	private:
		unsigned m_max_line_cnt{5};
		unsigned m_line_cnt{0};
		std::deque<LogDesc> m_logs;
	};
}
