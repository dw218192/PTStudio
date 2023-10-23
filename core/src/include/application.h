#pragma once
#include <sstream>

template <unsigned MaxLine>
struct Application {

    Application() = default;
    virtual ~Application() = default;

    virtual void run() = 0;
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
        if (m_line_cnt > MaxLine) {
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
    unsigned m_line_cnt = 0;
    std::string m_messages;
};