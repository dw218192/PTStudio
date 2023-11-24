#pragma once

#include <imgui.h>
#include <type_traits>

#include "reflection.h"

namespace ImGui {
    template<typename Reflected, typename = std::enable_if_t<PTS::is_reflectable<Reflected>::value>>
    bool ReflectedField(const char* label, Reflected& reflected, bool collapsed = false);
}


// reflectable types can be inspected in the editor and modified automatically
template<typename Reflected, typename>
bool ImGui::ReflectedField(const char* label, Reflected& reflected, bool collapsed) {
    using namespace PTS;
    
    bool changed = false;
    if (ImGui::CollapsingHeader(label, collapsed ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
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
            } else if constexpr (std::is_enum_v<FieldType>) {
                if (auto enum_mod = field_info.template get_modifier<MEnum>()) {
                    auto&& field = field_info.get(reflected);
                    if (ImGui::Combo(field_info.var_name.data(), reinterpret_cast<int*>(&field), 
                        MEnum::imgui_callback_adapter,
                        &enum_mod,
                        enum_mod->num_items)
                    ) {
                        changed = true;
                    }
                } else {
                    ImGui::Text("Enum type %s does not have a modifier", field_info.type_name.data());
                }
            } else {
                ImGui::Text("Unsupported type: %s", field_info.type_name.data());
            }
        });
    }
    return changed;
}