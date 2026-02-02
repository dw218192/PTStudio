#include <core/imgui/fileDialogue.h>
#include <portable-file-dialogs.h>

auto ImGui::FileDialogue(FileDialogueMode mode, const std::vector<std::string>& filters,
                         const std::string& default_path) -> std::string {
    if (mode == FileDialogueMode::Open) {
        auto selection = pfd::open_file("Open File", default_path, filters).result();
        if (!selection.empty()) {
            return selection[0];
        }
    } else {
        auto result = pfd::save_file("Save File", default_path, filters).result();
        if (!result.empty()) {
            return result;
        }
    }

    return {};
}
