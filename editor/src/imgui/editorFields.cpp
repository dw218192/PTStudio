#include "../include/imgui/editorFields.h"

#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>

bool ImGui::TransformField(const char* label, Transform& transform, ImGuizmo::OPERATION& op, ImGuizmo::MODE& mode, bool& snap, glm::vec3& snap_scale) {
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
        
        if(ImGui::Checkbox("snap to grid", &snap)) {
            changed = true;
        }

        if (snap) {
            if(ImGui::InputFloat3("snap distance", glm::value_ptr(snap_scale))) {
                changed = true;
                snap_scale = glm::clamp(snap_scale, glm::vec3(0.02f), glm::vec3(100.0f));
            }
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
    }
    return changed;
}

bool ImGui::MaterialField(const char* label, Material& material) {
    bool changed = false;
    if(ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
        Material::for_each_field([&](auto field) {
            if constexpr (std::is_same_v<typename decltype(field)::type, glm::vec3>) {
                changed |= ImGui::ColorEdit3(field.var_name.data(), glm::value_ptr(field.get(material)));
            } else if constexpr (std::is_same_v<typename decltype(field)::type, float>) {
                changed |= ImGui::SliderFloat(field.var_name.data(), &field.get(material), 0, 2.0f);
            }
        });
    }
    return changed;
}

bool ImGui::ShaderVariableField(const char* label, UniformVar& variable) {
    bool changed = false;
    switch (variable.get_type()) {
        case ShaderVariableType::Mat3: {
            ImGui::Text("%s (Mat3)", label);

            auto value = variable.get_value<glm::mat3>().get();
            for (int i = 0; i < 3; ++i) {
                if (changed |= ImGui::InputFloat3(std::to_string(i).c_str(), glm::value_ptr(value[i]))) {
                    variable.set_value(value);
                }
            }
            break;
        }
        case ShaderVariableType::Mat4:{
            ImGui::Text("%s (Mat4)", label);

            auto value = variable.get_value<glm::mat4>().get();
            for (int i = 0; i < 4; ++i) {
                if (changed |= ImGui::InputFloat4(std::to_string(i).c_str(), glm::value_ptr(value[i]))) {
                    variable.set_value(value);
                }
            }
            break;
        }
        case ShaderVariableType::Vec2: {
            ImGui::Text("%s (Vec2)", label);

            auto value = variable.get_value<glm::vec2>().get();
            if (changed |= ImGui::InputFloat2("value", glm::value_ptr(value))) {
                variable.set_value(value);
            }
            
            break;
        }
        case ShaderVariableType::Vec3:
            ImGui::Text("%s (Vec3)", label);

            auto value = variable.get_value<glm::vec3>().get();
            if (changed |= ImGui::InputFloat3("value", glm::value_ptr(value))) {
                variable.set_value(value);
            }
        
            break;
        case ShaderVariableType::Vec4: {
            ImGui::Text("%s (Vec4)", label);

            auto value = variable.get_value<glm::vec4>().get();
            if (changed |= ImGui::InputFloat4("value", glm::value_ptr(value))) {
                variable.set_value(value);
            }

            break;
        }
        case ShaderVariableType::Float: {
            ImGui::Text("%s (Float)", label);

            auto value = variable.get_value<float>().get();
            if(changed |= ImGui::InputFloat("value", &value)) {
                variable.set_value(value);
            }
            break;
        }
        case ShaderVariableType::Int: {
            ImGui::Text("%s (Int)", label);
            
            auto value = variable.get_value<int>().get();
            if(changed |= ImGui::InputInt("value", &value)) {
                variable.set_value(value);
            }   
            break;
        }
    }

    return changed;
}
