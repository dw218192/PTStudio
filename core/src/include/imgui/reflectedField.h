#pragma once

#include <imgui.h>
#include <type_traits>
#include <glm/glm.hpp>

#include "reflection.h"
#include "typeTraitsUtil.h"

namespace ImGui {
    template<typename Reflected, typename = std::enable_if_t<PTS::Traits::is_reflectable<Reflected>::value>>
    bool ReflectedField(const char* label, Reflected& reflected, bool collapsed = false);

    namespace detail {
        // tag types for dispatching
        struct EnumType {};
        struct ReflectedType {};
        struct ContainerType {};
        struct PairType {};
        struct TupleType {};

        template<typename T>
        struct Dispatch {
            using type = std::conditional_t<std::is_enum_v<T>, EnumType,
                std::conditional_t<PTS::Traits::is_reflectable<T>::value, ReflectedType, 
                std::conditional_t<PTS::Traits::is_container<T>::value, ContainerType,
                std::conditional_t<PTS::Traits::is_pair<T>::value, PairType,
                std::conditional_t<PTS::Traits::is_tuple<T>::value, TupleType,
                T>>>>>;
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

        // ---------------------- primitive types ----------------------
        template<>
        struct DoField<float> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                constexpr auto range_mod = field_info.template get_modifier<MRange<typename FieldInfo::type>>();
                if constexpr (range_mod) {
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
                constexpr auto range_mod = field_info.template get_modifier<MRange<typename FieldInfo::type>>();
                if constexpr (range_mod) {
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
        template<typename T>
        struct DoField<T*> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                ImGui::Text(
                    "%s::%s @ %p",
                    field_info.type_name.data(), 
                    field_info.var_name.data(), 
                    field_info.get(reflected)
                );
                if constexpr (std::is_base_of_v<PTS::Object, T>) {
                    if (field_info.get(reflected))
                        ImGui::Text("Pointed-to Type: %s", field_info.get(reflected)->dyn_get_class_info().class_name.data());
                }
                return true;
            }
        };
        template<typename T>
        struct DoField<T const*> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                return DoField<T*>::impl(field_info, reflected);
            }
        };
        
        template<>
        struct DoField<EnumType> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                bool changed = false;

                constexpr auto enum_mod = field_info.template get_modifier<MEnum>();
                constexpr auto enum_flags_mod = field_info.template get_modifier<MEnumFlags>();

                if constexpr (enum_mod) {
                    auto&& field = field_info.get(reflected);
                    auto field_int = reinterpret_cast<int*>(&field);

                    auto preview = enum_mod->num_items == 0 ? "None" : enum_mod->get_name(*field_int);
                    if (ImGui::BeginCombo(field_info.var_name.data(), preview)) {
                        for (int i = 0; i < enum_mod->num_items; ++i) {
                            bool is_selected = *field_int == i;
                            if (ImGui::Selectable(enum_mod->get_name(i), is_selected)) {
                                *field_int = i;
                                changed = true;
                            }
                        }
                        ImGui::EndCombo();
                    }
                } else if constexpr (enum_flags_mod) {
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

        // ---------------------- glm types ----------------------
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
                constexpr auto color_mod = field_info.template get_modifier<MColor>();
                if constexpr (color_mod) {
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
                constexpr auto color_mod = field_info.template get_modifier<MColor>()
                if constexpr (color_mod) {
                    return ImGui::ColorEdit4(field_info.var_name.data(), glm::value_ptr(field));
                } else {
                    return ImGui::InputFloat4(field_info.var_name.data(), glm::value_ptr(field));
                }
            }
        };

        // ---------------------- container types ----------------------
        template<>
        struct DoField<ContainerType> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                bool changed = false;
                if (ImGui::TreeNode(field_info.var_name.data())) {
                    if (ImGui::Button("Add")) {
                        field.emplace_back();
                        changed = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Remove")) {
                        if (!field.empty()) {
                            field.pop_back();
                            changed = true;
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Clear")) {
                        field.clear();
                        changed = true;
                    }
                    ImGui::Separator();
                    auto it = field.begin();
                    for (int i = 0; i < field.size(); ++i, ++it) {
                        ImGui::PushID(i);
                        if (ImGui::TreeNode(std::to_string(i).data())) {
                            // TODO: this is ugly
                            auto sub_var_name = std::string{ field_info.var_name.data() } + "[" + std::to_string(i) + "]";
                            auto sub_type_name = std::string{ field_info.type_name.data() } + "::value_type";
                            using ContainerType = typename FieldInfo::type;
                            using ElementType = typename ContainerType::value_type;

                            struct ContainerElementFieldInfo {
                                std::string_view type_name;
                                std::string_view var_name;
                                ContainerElementFieldInfo(std::string_view type_name, std::string_view var_name, ContainerType& container,  int i)
                                    : type_name{ type_name }, var_name{ var_name }, m_container{ container }, m_i{ i } {}
                                auto get(Reflected& reflected) -> ElementType& {
                                    (void) reflected;
                                    return m_container[m_i];
                                }
                            private:
                                ContainerType& m_container;
                                int m_i;
                            };

                            auto sub_field_info = ContainerElementFieldInfo{ sub_type_name, sub_var_name, field, i };
                            changed |= DoField<Dispatch_t<ElementType>>::impl(sub_field_info, reflected);
                            ImGui::TreePop();
                        }
                        ImGui::PopID();
                    }
                    ImGui::TreePop();
                }
                return changed;
            }
        };

        template<>
        struct DoField<PairType> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                bool changed = false;
                if (ImGui::TreeNode(field_info.var_name.data())) {
                    changed |= ImGui::ReflectedField("First", field.first);
                    changed |= ImGui::ReflectedField("Second", field.second);
                    ImGui::TreePop();
                }
                return changed;
            }
        };

        template<>
        struct DoField<TupleType> {
            template<typename Reflected, typename FieldInfo>
            static auto impl(FieldInfo field_info, Reflected& reflected) -> bool {
                auto&& field = field_info.get(reflected);
                bool changed = false;
                if (ImGui::TreeNode(field_info.var_name.data())) {
                    std::apply([&changed](auto&&... args) {
                        (changed |= ImGui::ReflectedField("Element", args), ...);
                    }, field);
                    ImGui::TreePop();
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
            if constexpr (!field_info.template has_modifier<MNoInspect>()) {
                auto old_val = field_info.get(reflected);
                if (detail::DoField<detail::Dispatch_t<FieldType>>::impl(field_info, reflected)) {
                    changed = true;
                    field_info.on_change(old_val, field_info.get(reflected), reflected);
                }
            }
        });
    }
    return changed;
}