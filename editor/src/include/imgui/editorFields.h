#pragma once

#include <imgui.h>
#include <ImGuizmo.h>

#include "transform.h"
#include "material.h"
#include "reflection.h"
#include "UniformVar.h"

namespace ImGui {    
    bool TransformField(const char* label, PTS::Transform& transform, ImGuizmo::OPERATION& op, ImGuizmo::MODE& mode, bool& snap, glm::vec3& snap_scale);
    bool ShaderVariableField(const char* label, PTS::UniformVar& variable);
    
    // reflectable types can be inspected in the editor and modified automatically
    template<typename Reflected, typename = std::enable_if_t<PTS::is_reflectable<Reflected>::value>>
    bool ReflectedField(const char* label, Reflected& reflected) {
        using namespace PTS;
        bool changed = false;
        if (ImGui::CollapsingHeader(label)) {
            Reflected::for_each_field([&](auto field_info) {
                using FieldType = typename decltype(field_info)::type;
                if (auto no_inpsect_mod = field_info.template get_modifier<MNoInspect>()) {
                    return;
                }

                if constexpr (std::is_same_v<FieldType, float>) {
                    auto&& field = field_info.get(reflected);
                    if (auto range_mod = field_info.template get_modifier<MRange<FieldType>>()) {
                        if (ImGui::SliderFloat(field_info.var_name.data(), &field, range_mod->min, range_mod->max)) {
                            changed = true;
                        }
                    } else if (auto min_mod = field_info.template get_modifier<MMin<FieldType>>()) {
                        if (ImGui::SliderFloat(field_info.var_name.data(), &field, min_mod->value, 100000.0f)) {
                            changed = true;
                        }
                    } else if (auto max_mod = field_info.template get_modifier<MMax<FieldType>>()) {
                        if (ImGui::SliderFloat(field_info.var_name.data(), &field, -100000.0f, max_mod->value)) {
                            changed = true;
                        }
                    } else {
                        if (ImGui::InputFloat(field_info.var_name.data(), &field)) {
                            changed = true;
                        }
                    }
                } else if constexpr (std::is_same_v<FieldType, int>) {
                    auto&& field = field_info.get(reflected);
                    if (auto range_mod = field_info.template get_modifier<MRange<FieldType>>()) {
                        if (ImGui::SliderInt(field_info.var_name.data(), &field, range_mod->min, range_mod->max)) {
                            changed = true;
                        }
                    } else if (auto min_mod = field_info.template get_modifier<MMin<FieldType>>()) {
                        if (ImGui::SliderInt(field_info.var_name.data(), &field, min_mod->value)) {
                            changed = true;
                        }
                    } else if (auto max_mod = field_info.template get_modifier<MMax<FieldType>>()) {
                        if (ImGui::SliderInt(field_info.var_name.data(), &field, 0, max_mod->value)) {
                            changed = true;
                        }
                    } else {
                        if (ImGui::InputInt(field_info.var_name.data(), &field)) {
                            changed = true;
                        }
                    }
                } else if constexpr (std::is_same_v<FieldType, bool>) {
                    auto&& field = field_info.get(reflected);
                    if (ImGui::Checkbox(field_info.var_name.data(), &field)) {
                        changed = true;
                    }
                } else if constexpr (std::is_same_v<FieldType, glm::vec2>) {
                    auto&& field = field_info.get(reflected);
                    if (ImGui::InputFloat2(field_info.var_name.data(), glm::value_ptr(field))) {
                        changed = true;
                    }
                } else if constexpr (std::is_same_v<FieldType, glm::vec3>) {
                    auto&& field = field_info.get(reflected);
                    if (auto color_mod = field_info.template get_modifier<MColor>()) {
                        if (ImGui::ColorEdit3(field_info.var_name.data(), glm::value_ptr(field))) {
                            changed = true;
                        }
                    } else {
                        if (ImGui::InputFloat3(field_info.var_name.data(), glm::value_ptr(field))) {
                            changed = true;
                        }
                    }
                } else if constexpr (std::is_same_v<FieldType, glm::vec4>) {
                    auto&& field = field_info.get(reflected);
                    if (auto color_mod = field_info.template get_modifier<MColor>()) {
                        if (ImGui::ColorEdit4(field_info.var_name.data(), glm::value_ptr(field))) {
                            changed = true;
                        }
                    } else {
                        if (ImGui::InputFloat4(field_info.var_name.data(), glm::value_ptr(field))) {
                            changed = true;
                        }
                    }
                } else {
                    ImGui::Text("Unsupported type: %s", field_info.type_name.data());
                }
            });
        }
        return changed;
    }
}