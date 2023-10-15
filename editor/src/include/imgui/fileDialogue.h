#pragma once

#include <string_view>

namespace ImGui {
    auto FileDialogue(const char* filter = nullptr, const char* defaultPath = nullptr) -> std::string;
}