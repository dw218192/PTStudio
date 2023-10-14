#pragma once
#include <sstream>

struct EditorConsole {
    EditorConsole();
    EditorConsole(EditorConsole const&) = delete;
    EditorConsole(EditorConsole&&) = delete;
    EditorConsole& operator=(EditorConsole const&) = delete;
    EditorConsole& operator=(EditorConsole&&) = delete;
    ~EditorConsole();

private:
    std::stringstream m_stream;
};