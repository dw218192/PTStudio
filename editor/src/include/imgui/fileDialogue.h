#pragma once

#include <string_view>

namespace ImGui {
    bool FileDialogue(const char* label, std::string_view path, const char* filter = nullptr, const char* defaultPath = nullptr);
}