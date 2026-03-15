#pragma once

#include <functional>
#include <string>
#include <vector>

namespace ImGui {

enum class FileDialogueMode { Open, Save };

/// Result delivered to the callback: filename + file contents.
struct FileDialogueResult {
    std::string name;
    std::string contents;
};

/// Async file dialog — works on all platforms including Emscripten.
/// On native: blocks, reads the file, invokes callback before returning.
/// On Emscripten: triggers browser file picker, callback fires later.
/// The accept filter is a MIME type or extension string (e.g. ".usda,.usdc,.usd").
void FileDialogueAsync(FileDialogueMode mode, const std::string& accept,
                       std::function<void(FileDialogueResult)> on_result);

}  // namespace ImGui