#pragma once

#include <imgui.h>

#include <glm/glm.hpp>
#include <type_traits>

#include "reflection.h"
#include "typeTraitsUtil.h"

namespace ImGui {
template <typename TemplatedFieldInfo, typename Reflected>
auto Field(char const* label, TemplatedFieldInfo field_info, Reflected& reflected)
    -> std::enable_if_t<PTS::Traits::has_tfield_interface_v<TemplatedFieldInfo>, bool>;

template <typename Reflected>
auto ReflectedField(char const* label, Reflected& reflected, bool collapsed = false)
    -> std::enable_if_t<PTS::Traits::is_reflectable_v<Reflected>, bool>;

namespace detail {
// tag types for dispatching
struct EnumType {};

struct ReflectedType {};

struct ContainerType {};

struct TupleLikeType {};

template <typename T>
struct Dispatch {
    using type = std::conditional_t<
        std::is_enum_v<T>, EnumType,
        std::conditional_t<PTS::Traits::is_reflectable_v<T>, ReflectedType,
                           std::conditional_t<PTS::Traits::is_container_v<T>, ContainerType,
                                              std::conditional_t<PTS::Traits::is_tuple_like_v<T>,
                                                                 TupleLikeType, T>>>>;
};

template <typename T>
using Dispatch_t = typename Dispatch<T>::type;

template <typename T>
struct DoField {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        ImGui::Text("Unsupported type: %s", field_info.type_name.data());
        return false;
    }
};

template <>
struct DoField<ReflectedType> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto&& field = field_info.get(reflected);
        return ImGui::ReflectedField(label, field);
    }
};

// ---------------------- primitive types ----------------------
template <>
struct DoField<float> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto&& field = field_info.get(reflected);
        // TODO: fix constexpr
        auto range_mod = field_info.template get_modifier<PTS::MRange<typename FieldInfo::type>>();
        if (range_mod) {
            return ImGui::SliderFloat(label, &field, range_mod->min, range_mod->max);
        } else {
            return ImGui::InputFloat(label, &field);
        }
    }
};

template <>
struct DoField<int> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto&& field = field_info.get(reflected);
        // TODO: fix constexpr
        auto range_mod = field_info.template get_modifier<PTS::MRange<typename FieldInfo::type>>();
        if (range_mod) {
            return ImGui::SliderInt(label, &field, range_mod->min, range_mod->max);
        } else {
            return ImGui::InputInt(label, &field);
        }
    }
};

template <>
struct DoField<bool> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto&& field = field_info.get(reflected);
        return ImGui::Checkbox(label, &field);
    }
};

template <typename T>
struct DoField<T*> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        ImGui::Text("%s %s @ %p", field_info.type_name.data(), field_info.var_name.data(),
                    field_info.get(reflected));
        if constexpr (std::is_base_of_v<PTS::Object, T>) {
            if (field_info.get(reflected))
                ImGui::Text("Pointed-to Type: %s",
                            field_info.get(reflected)->dyn_get_class_info().class_name.data());
        }
        return true;
    }
};

template <typename T>
struct DoField<T const*> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        return DoField<T*>::impl(label, field_info, reflected);
    }
};

template <>
struct DoField<EnumType> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto changed = false;

        // TODO: fix constexpr
        auto enum_mod = field_info.template get_modifier<PTS::MEnum>();
        auto enum_flags_mod = field_info.template get_modifier<PTS::MEnumFlags>();

        if (enum_mod) {
            auto&& field = field_info.get(reflected);
            auto field_int = reinterpret_cast<int*>(&field);

            auto preview = enum_mod->num_items == 0 ? "None" : enum_mod->get_name(*field_int);
            if (ImGui::BeginCombo(label, preview)) {
                for (auto i = 0; i < enum_mod->num_items; ++i) {
                    auto is_selected = *field_int == i;
                    if (ImGui::Selectable(enum_mod->get_name(i), is_selected)) {
                        *field_int = i;
                        changed = true;
                    }
                }
                ImGui::EndCombo();
            }
        } else if (enum_flags_mod) {
            auto&& field = field_info.get(reflected);
            auto const field_int = reinterpret_cast<int*>(&field);

            std::string preview = *field_int == 0 ? "None" : "";
            for (auto i = 0; i < enum_flags_mod->num_items; ++i) {
                if (*field_int & (1 << i)) {
                    preview += enum_flags_mod->get_name(i);
                    if (i < enum_flags_mod->num_items - 1) preview.push_back(',');
                }
            }
            if (ImGui::BeginCombo(label, preview.c_str())) {
                if (ImGui::Selectable("None", *field_int == 0)) {
                    *field_int = 0;
                    changed = true;
                }
                for (auto i = 0; i < enum_flags_mod->num_items; ++i) {
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
template <>
struct DoField<glm::vec2> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto&& field = field_info.get(reflected);
        return ImGui::InputFloat2(label, glm::value_ptr(field));
    }
};

template <>
struct DoField<glm::vec3> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto&& field = field_info.get(reflected);
        // TODO: fix constexpr
        auto color_mod = field_info.template get_modifier<PTS::MColor>();
        if (color_mod) {
            return ImGui::ColorEdit3(label, glm::value_ptr(field));
        } else {
            return ImGui::InputFloat3(label, glm::value_ptr(field));
        }
    }
};

template <>
struct DoField<glm::vec4> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        auto&& field = field_info.get(reflected);
        // TODO: fix constexpr
        auto color_mod = field_info.template get_modifier<PTS::MColor>();
        if (color_mod) {
            return ImGui::ColorEdit4(label, glm::value_ptr(field));
        } else {
            return ImGui::InputFloat4(label, glm::value_ptr(field));
        }
    }
};

// ---------------------- container types ----------------------
// adaptors to satisfy the interface
template <typename ClassType, typename ElementType, typename ContainerType>
struct ContainerElementFieldInfo {
    using type = ElementType;

    std::string_view type_name;
    std::string_view var_name;

    ContainerElementFieldInfo(std::string_view type_name, std::string_view var_name,
                              ContainerType& container, int i)
        : type_name{type_name}, var_name{var_name}, m_container{container}, m_i{i} {
    }

    auto get(ClassType&) -> auto& {
        return m_container[m_i];
    }

    template <typename Mod>
    constexpr auto has_modifier() {
        return false;
    }

    template <typename Mod>
    constexpr auto get_modifier() -> Mod const* {
        return nullptr;
    }

   private:
    ContainerType& m_container;
    int m_i;
};

template <typename ClassType, typename ElementType, typename TupleType, int I>
struct TupleElementFieldInfo {
    using type = std::tuple_element_t<I, TupleType>;

    std::string_view type_name;
    std::string_view var_name;

    TupleElementFieldInfo(std::string_view type_name, std::string_view var_name, TupleType& tuple)
        : type_name{type_name}, var_name{var_name}, m_tuple{tuple} {
    }

    auto get(ClassType& reflected) -> auto& {
        return std::get<I>(m_tuple);
    }

    template <typename Mod>
    constexpr auto has_modifier() {
        return false;
    }

    template <typename Mod>
    constexpr auto get_modifier() -> Mod const* {
        return nullptr;
    }

   private:
    TupleType& m_tuple;
};

template <>
struct DoField<ContainerType> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        using ContainerType = typename FieldInfo::type;
        using ElementType = typename ContainerType::value_type;

        auto&& field = field_info.get(reflected);
        auto changed = false;
        if (ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
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

            auto it = field.begin();
            for (auto i = 0; i < field.size(); ++i, ++it) {
                ImGui::Separator();
                auto sub_var_name =
                    std::string{field_info.var_name.data()} + "[" + std::to_string(i) + "]";
                auto sub_type_name = Type::of<ElementType>().to_string();
                auto sub_field_info =
                    ContainerElementFieldInfo<Reflected, ElementType, ContainerType>{
                        sub_type_name, sub_var_name, field, i};
                changed |= DoField<detail::Dispatch_t<ElementType>>::impl(
                    sub_var_name.data(), sub_field_info, reflected);
            }
        }
        return changed;
    }
};

template <>
struct DoField<TupleLikeType> {
    template <typename Reflected, typename FieldInfo>
    static auto impl(char const* label, FieldInfo field_info, Reflected& reflected) -> bool {
        using TupleType = typename FieldInfo::type;

        auto changed = false;
        auto&& field = field_info.get(reflected);

        if (ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
            PTS::Traits::for_each::element_in_tuple(
                field, [&changed, &reflected, &field_info, &field](auto&& arg) {
                    ImGui::Separator();

                    // this will return the first index of the type in the tuple
                    // TODO: make this work for duplicate types in the
                    // tuple?
                    constexpr auto I = PTS::Traits::find_v<TupleType, std::decay_t<decltype(arg)>>;
                    auto sub_var_name =
                        std::string{field_info.var_name.data()} + "[" + std::to_string(I) + "]";
                    auto sub_type_name =
                        PTS::Type::of<std::tuple_element_t<I, TupleType>>().to_string();
                    auto sub_field_info =
                        TupleElementFieldInfo<Reflected, std::tuple_element_t<I, TupleType>,
                                              TupleType, I>{sub_type_name, sub_var_name, field};

                    changed |= DoField<Dispatch_t<std::tuple_element_t<I, TupleType>>>::impl(
                        sub_var_name.c_str(), sub_field_info, reflected);
                });
        }
        return changed;
    }
};
}  // namespace detail

template <typename TemplatedFieldInfo, typename Reflected>
auto Field(char const* label, TemplatedFieldInfo field_info, Reflected& reflected)
    -> std::enable_if_t<PTS::Traits::has_tfield_interface_v<TemplatedFieldInfo>, bool> {
    auto changed = false;
    using FieldType = typename TemplatedFieldInfo::type;
    if constexpr (!field_info.template has_modifier<PTS::MNoInspect>()) {
        if constexpr (field_info.template has_modifier<PTS::MReadOnly>()) {
            ImGui::BeginDisabled();
        }
        if (detail::DoField<detail::Dispatch_t<FieldType>>::impl(label, field_info, reflected)) {
            changed = true;
            field_info.on_change(field_info.get(reflected), reflected);
        }
        if constexpr (field_info.template has_modifier<PTS::MReadOnly>()) {
            ImGui::EndDisabled();
        }
    }
    return changed;
}

// reflectable types can be inspected in the editor and modified automatically
template <typename Reflected>
auto ReflectedField(char const* label, Reflected& reflected, bool collapsed)
    -> std::enable_if_t<PTS::Traits::is_reflectable_v<Reflected>, bool> {
    auto changed = false;
    if (CollapsingHeader(label, collapsed ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
        Reflected::for_each_field([&changed, &reflected](auto field_info) {
            changed |= Field(field_info.var_name.data(), field_info, reflected);
        });
    }
    return changed;
}
}  // namespace ImGui
