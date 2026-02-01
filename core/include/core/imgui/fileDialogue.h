#pragma once

#include <string>
#include <vector>

namespace ImGui {

enum class FileDialogueMode { Open, Save };

// File filters as pairs: {"Description", "*.ext1 *.ext2", "Other", "*.ext3", ...}
// Example: {"Image Files", "*.png *.jpg *.gif", "All Files", "*"}
auto FileDialogue(FileDialogueMode mode,
                  const std::vector<std::string>& filters = {"All Files", "*"},
                  const std::string& default_path = {}) -> std::string;

}  // namespace ImGui