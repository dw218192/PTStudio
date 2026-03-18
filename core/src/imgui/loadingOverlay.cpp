#include <core/imgui/loadingOverlay.h>
#include <imgui.h>

#include <algorithm>

namespace pts {

void LoadingOverlay::track(TrackedTask task) {
    m_tasks.push_back(std::move(task));
}

bool LoadingOverlay::draw() {
    // Remove completed tasks
    m_tasks.erase(std::remove_if(m_tasks.begin(), m_tasks.end(),
                                 [](const TrackedTask& t) { return t.is_done(); }),
                  m_tasks.end());

    if (m_tasks.empty()) {
        return false;
    }

    auto& io = ImGui::GetIO();

    // Input blocker — drawn first so it's behind the progress window in z-order
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("##LoadingBlocker", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);
    ImGui::InvisibleButton("##block", io.DisplaySize);
    ImGui::End();

    // Fullscreen semi-transparent overlay
    ImGui::GetForegroundDrawList()->AddRectFilled(ImVec2(0, 0), io.DisplaySize,
                                                  IM_COL32(0, 0, 0, 128));

    // Centered progress window
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin("##LoadingOverlay", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoNav);

    for (size_t i = 0; i < m_tasks.size(); ++i) {
        auto& task = m_tasks[i];
        ImGui::Text("%s", task.name.c_str());
        ImGui::ProgressBar(task.progress(), ImVec2(300, 0));
        auto status_text = task.status();
        if (!status_text.empty()) {
            ImGui::TextDisabled("%s", status_text.c_str());
        }
        if (i + 1 < m_tasks.size()) {
            ImGui::Separator();
        }
    }

    ImGui::End();

    return true;
}

bool LoadingOverlay::has_active_tasks() const {
    return !m_tasks.empty();
}

}  // namespace pts
