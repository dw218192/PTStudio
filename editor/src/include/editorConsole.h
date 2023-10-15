#pragma once

#include <sstream>
#include <deque>
#include <string_view>

template<unsigned MaxLine>
struct EditorConsole {
    auto str() noexcept -> std::string_view {
        return m_messages;
    }
    template<typename... Args>
    void log(Args&&... args) noexcept {
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
    }

private:
    unsigned m_line_cnt = 0;
    std::string m_messages;
};