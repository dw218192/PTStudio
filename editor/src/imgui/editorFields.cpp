#include "../include/imgui/editorFields.h"

#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>

bool ImGui::TransformField(const char* label, Transform& transform, ImGuizmo::OPERATION& op, ImGuizmo::MODE& mode, bool& snap) {
    bool changed = false;
    if(ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto pos = transform.get_position();
        auto rot = transform.get_rotation();
        auto scale = transform.get_scale();

        if(ImGui::InputFloat3("position", glm::value_ptr(pos))) {
            changed = true;
            transform.set_position(TransformSpace::WORLD, pos);
        }
        if(ImGui::InputFloat3("rotation", glm::value_ptr(rot))) {
            changed = true;
            transform.set_rotation(TransformSpace::WORLD, rot);
        }
        if(ImGui::InputFloat3("scale", glm::value_ptr(scale))) {
            changed = true;
            transform.set_scale(TransformSpace::WORLD, scale);
        }

        ImGui::SeparatorText("Transform Mode");
        if (ImGui::RadioButton("Translate", op == ImGuizmo::TRANSLATE)) {
            changed = true;
            op = ImGuizmo::TRANSLATE;
        }
        if (ImGui::RadioButton("Rotate", op == ImGuizmo::ROTATE)) {
            changed = true;
            op = ImGuizmo::ROTATE;
        }
        if (ImGui::RadioButton("Scale", op == ImGuizmo::SCALE)) {
            changed = true;
        	op = ImGuizmo::SCALE;
        }
        
        ImGui::SeparatorText("Transform Space");
        if (op != ImGuizmo::SCALE) {
            if (ImGui::RadioButton("Local", mode == ImGuizmo::LOCAL)) {
                changed = true;
                mode = ImGuizmo::LOCAL;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("World", mode == ImGuizmo::WORLD)) {
                changed = true;
                mode = ImGuizmo::WORLD;
            }
        }
        
        changed |= ImGui::Checkbox("snap to grid", &snap);
    }
    return changed;
}
