#include "../include/imgui/fileDialogue.h"

#include <imgui.h>
#include <nfd.h>


auto ImGui::FileDialogue(FileDialogueMode mode, const char* filter, const char* defaultPath) -> std::string {
    nfdchar_t* outPath = nullptr;
    nfdresult_t result;
    if (mode == FileDialogueMode::OPEN) {
        result = NFD_OpenDialog(filter, defaultPath, &outPath);
    } else {
        result = NFD_SaveDialog(filter, defaultPath, &outPath);
    }

    if (result == NFD_OKAY) {
	    std::string path = outPath;
        free(outPath);
        return path;
    }
    else if (result == NFD_CANCEL) {
        return {};
    }
    else {
        ImGui::OpenPopup("Error");
    }
    if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Error opening file dialogue");
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    return {};
}
