#include <core/imgui/loadingOverlay.h>
#include <imgui.h>

#include <algorithm>
#include <string>

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
                                                  IM_COL32(0, 0, 0, 100));

    // Centered progress window — fixed width
    constexpr float k_window_width = 360.0f;
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(k_window_width, 0));
    ImGui::SetNextWindowBgAlpha(0.9f);
    ImGui::Begin("##LoadingOverlay", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

    float const content_width = ImGui::GetContentRegionAvail().x;

    for (size_t i = 0; i < m_tasks.size(); ++i) {
        auto& task = m_tasks[i];
        ImGui::Text("%s", task.name.c_str());
        ImGui::ProgressBar(task.progress(), ImVec2(content_width, 0));
        auto status_text = task.status();
        if (!status_text.empty()) {
            // Middle-ellipsis: if text is too wide, show "start...end"
            float max_w = content_width;
            float text_w = ImGui::CalcTextSize(status_text.c_str()).x;
            if (text_w > max_w && status_text.size() > 10) {
                auto ellipsis = std::string("...");
                float ellipsis_w = ImGui::CalcTextSize(ellipsis.c_str()).x;
                float budget = max_w - ellipsis_w;
                // Binary search is overkill — take a fixed prefix/suffix ratio
                size_t total = status_text.size();
                size_t suffix_len = total / 3;
                size_t prefix_len = total - suffix_len;
                // Shrink until it fits
                while (prefix_len > 1 && suffix_len > 1) {
                    auto candidate = status_text.substr(0, prefix_len) + ellipsis +
                                     status_text.substr(total - suffix_len);
                    if (ImGui::CalcTextSize(candidate.c_str()).x <= max_w) {
                        status_text = std::move(candidate);
                        break;
                    }
                    prefix_len -= 1;
                    suffix_len -= 1;
                }
            }
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
