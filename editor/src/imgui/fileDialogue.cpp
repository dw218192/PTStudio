#include "../include/imgui/fileDialogue.h"

#include <imgui.h>
#include <nfd.h>

bool ImGui::FileDialogue(const char* label, std::string_view path, const char* filter, const char* defaultPath) {
    if (ImGui::Button(label)) {
        nfdchar_t* outPath = nullptr;
        nfdresult_t result = NFD_OpenDialog(filter, defaultPath, &outPath);
        if (result == NFD_OKAY) {
            path = outPath;
            free(outPath);
            return true;
        }
        else if (result == NFD_CANCEL) {
            return false;
        }
        else {
            ImGui::OpenPopup("Error");
        }
    }
    if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Error opening file dialogue");
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    return false;
}
