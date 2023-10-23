#pragma once
#include <sstream>

struct Application {
    Application() = default;
    virtual ~Application() = default;
    virtual void run() = 0;

    void set_max_log_cnt(unsigned cnt) { m_max_line_cnt = cnt; }

    /**
     * \brief Terminates the program with the given exit code
     * \param code the exit code
    */
    [[noreturn]] virtual void quit(int code) = 0;

    template <typename... Args>
	void log(Args&&... args) {
        std::ostringstream ss;
        (ss << ... << args);
        ss << '\n';

        ++m_line_cnt;
        if (m_line_cnt > m_max_line_cnt) {
            if(!m_messages.empty())
				m_messages = m_messages.substr(m_messages.find('\n') + 1);
        	--m_line_cnt;
        }
        m_messages.append(ss.str());
        print(m_messages);
    }

protected:
    virtual void print(std::string_view msg) = 0;

private:
    unsigned m_line_cnt{ 0 };
    unsigned m_max_line_cnt{ 5 };
    std::string m_messages;
};