#pragma once

#include <imgui.h>
#include <type_traits>
#include <glm/glm.hpp>

#include "reflection.h"

namespace ImGui {
    template<typename Reflected, typename = std::enable_if_t<PTS::is_reflectable<Reflected>::value>>
    bool ReflectedField(const char* label, Reflected& reflected, bool collapsed = false);

    namespace detail {
        // tag types for dispatching
        struct EnumType {};
        struct ReflectedType {};

        template<typename T>
        struct Dispatch {
            using type = std::conditional_t<std::is_enum_v<T>, EnumType,
                std::conditional_t<PTS::is_reflectable<T>::value, ReflectedType, T>>;
        };
        template<typename T>
        using Dispatch_t = typename Dispatch<T>::type;

        template<typename T>
        struct DoField {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                ImGui::Text("Unsupported type: %s", field_info.type_name.data());
                return false;
            }
        };

        template<>
        struct DoField<ReflectedType> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                return ImGui::ReflectedField(field_info.var_name.data(), field);
            }
        };

        template<>
        struct DoField<float> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                if (auto range_mod = field_info.template get_modifier<MRange<typename FieldInfo::type>>()) {
                    return ImGui::SliderFloat(field_info.var_name.data(), &field, range_mod->min, range_mod->max);
                } else {
                    return ImGui::InputFloat(field_info.var_name.data(), &field);
                }
            }
        };

        template<>
        struct DoField<int> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                if (auto range_mod = field_info.template get_modifier<MRange<typename FieldInfo::type>>()) {
                    return ImGui::SliderInt(field_info.var_name.data(), &field, range_mod->min, range_mod->max);
                } else {
                    return ImGui::InputInt(field_info.var_name.data(), &field);
                }
            }
        };

        template<>
        struct DoField<bool> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                return ImGui::Checkbox(field_info.var_name.data(), &field);
            }
        };

        template<>
        struct DoField<glm::vec2> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                return ImGui::InputFloat2(field_info.var_name.data(), glm::value_ptr(field));
            }
        };

        template<>
        struct DoField<glm::vec3> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                if (auto color_mod = field_info.template get_modifier<MColor>()) {
                    return ImGui::ColorEdit3(field_info.var_name.data(), glm::value_ptr(field));
                } else {
                    return ImGui::InputFloat3(field_info.var_name.data(), glm::value_ptr(field));
                }
            }
        };

        template<>
        struct DoField<glm::vec4> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                if (auto color_mod = field_info.template get_modifier<MColor>()) {
                    return ImGui::ColorEdit4(field_info.var_name.data(), glm::value_ptr(field));
                } else {
                    return ImGui::InputFloat4(field_info.var_name.data(), glm::value_ptr(field));
                }
            }
        };

        template<>
        struct DoField<EnumType> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                bool changed = false;
                if (auto enum_mod = field_info.template get_modifier<MEnum>()) {
                    auto&& field = field_info.get(reflected);
                    changed = ImGui::Combo(field_info.var_name.data(), reinterpret_cast<int*>(&field), 
                        MEnum::imgui_callback_adapter,
                        &enum_mod,
                        enum_mod->num_items);
                } else if (auto enum_flags_mod = field_info.template get_modifier<MEnumFlags>()) {
                    auto&& field = field_info.get(reflected);
                    auto field_int = reinterpret_cast<int*>(&field);

                    std::string preview = *field_int == 0 ? "None" : "";
                    for (int i = 0; i < enum_flags_mod->num_items; ++i) {
                        if (*field_int & (1 << i)) {
                            preview += enum_flags_mod->get_name(i);
                            if (i < enum_flags_mod->num_items - 1)
                                preview.push_back(',');
                        }
                    }
                    if (ImGui::BeginCombo(field_info.var_name.data(), preview.c_str())) {
                        if (ImGui::Selectable("None", *field_int == 0)) {
                            *field_int = 0;
                            changed = true;
                        }
                        for (int i = 0; i < enum_flags_mod->num_items; ++i) {
                            bool is_selected = *field_int & (1 << i);
                            if (ImGui::Selectable(enum_flags_mod->get_name(i), is_selected)) {
                                *field_int ^= (1 << i);
                                changed = true;
                            }
                        }
                        ImGui::EndCombo();
                    }
                } else {
                    ImGui::Text("Enum type %s does not have a modifier", field_info.type_name.data());
                }
                return changed;
            }
        };

    } // namespace detail

} // namespace ImGui

// reflectable types can be inspected in the editor and modified automatically
template<typename Reflected, typename>
bool ImGui::ReflectedField(const char* label, Reflected& reflected, bool collapsed) {
    using namespace PTS;
    
    bool changed = false;
    if (ImGui::CollapsingHeader(label, collapsed ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
        Reflected::for_each_field([&changed, &reflected](auto field_info) {
            using FieldType = typename decltype(field_info)::type;
            if (auto no_inpsect_mod = field_info.template get_modifier<MNoInspect>()) {
                return;
            }
            auto old_val = field_info.get(reflected);
            if (detail::DoField<detail::Dispatch_t<FieldType>>::impl(field_info, reflected)) {
                changed = true;
                field_info.on_change(old_val, field_info.get(reflected), reflected);
            }
        });
    }
    return changed;
}