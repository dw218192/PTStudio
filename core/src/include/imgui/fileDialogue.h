#pragma once

#include <string>
#include <string_view>

namespace ImGui {
enum class FileDialogueMode { OPEN, SAVE };
auto FileDialogue(FileDialogueMode mode, const char* filter = nullptr,
                  const char* defaultPath = nullptr) -> std::string;
}  // namespace ImGui