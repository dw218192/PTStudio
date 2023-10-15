#include "../include/imgui/editorFields.h"

#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>

bool ImGui::TransformField(const char* label, Transform& transform) {
    bool changed = false;
    if(ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto pos = transform.get_position();
        auto rot = transform.get_rotation();
        auto scale = transform.get_scale();

        if(changed |= ImGui::DragFloat3("position", glm::value_ptr(pos), 0.1f)) {
            transform.set_position(TransformSpace::WORLD, pos);
        }
        if(changed |= ImGui::DragFloat3("rotation", glm::value_ptr(rot), 0.1f)) {
            transform.set_rotation(TransformSpace::WORLD, rot);
        }
        if(changed |= ImGui::DragFloat3("scale", glm::value_ptr(scale), 0.1f)) {
            transform.set_scale(TransformSpace::WORLD, scale);
        }
    }
    return changed;
}
