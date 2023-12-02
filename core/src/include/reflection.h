#pragma once
#include <string_view>
#include <utility>
#include <tuple>
#include <optional>
#include <vector>
#include <functional>

namespace PTS {
    /**
     * simple reflection framework
     * limitations:
     *  - only single declaration per line
     *  - FIELD(type<a,b,c,..>, name) doesn't work, alias the type first
     *  - relies on non-standard __COUNTER__ macro, but g++, clang, and msvc all support it
     *  - access control change in END_REFLECT() may unintentionally expose private members
    */

    /**
     * a modifier is an attribute that can be attached to a field
     * modifiers are used to change the behavior of the inspector and the serialization process
     * modifiers are optional
    */
    template<typename... Modifiers>
    struct ModifierPack {
        std::tuple<Modifiers...> args;
        constexpr ModifierPack(Modifiers... args) : args(std::make_tuple(args...)) {}

        template<typename Callable>
        constexpr void for_each(Callable&& callable) const {
            std::apply([&callable](auto&&... args) {
                (callable(args), ...);
            }, args);
        }
    };
    /**
     * @brief displays the field as a slider, only works for float and int
     * @tparam T the type of the field
    */
    template<typename T>
    struct MRange {
        T min, max;
        static constexpr std::string_view name = "range";
        constexpr MRange(T min, T max) : min(min), max(max) {}
    };
    /**
     * @brief mark the field as serializable, only fields with this modifier will be saved to the scene file
    */
    struct MSerialize {
        static constexpr std::string_view name = "serialize";
    };
    /**
     * @brief don't show this field in the inspector, can be used to hide explicitly handled fields or fields that shouldn't be edited
    */
    struct MNoInspect {
        static constexpr std::string_view name = "no inspect";
    };
    /**
     * @brief displays a color picker, only works for glm::vec3 and glm::vec4
    */
    struct MColor {
        static constexpr std::string_view name = "color";
    };
    /**
     * @brief displays an enum as a dropdown
    */
    struct MEnum {
        static constexpr std::string_view name = "enum";
        static auto imgui_callback_adapter(void* data, int idx, char const** out_text) -> bool {
            auto enum_info = reinterpret_cast<MEnum const*>(data);
            *out_text = enum_info->get_name(idx);
            return true;
        }
        int num_items;
        auto (*get_name)(int idx) -> char const*;
    };
    /**
     * @brief displays a multi-select enum as a dropdown
    */
    struct MEnumFlags {
        static constexpr std::string_view name = "enum flags";
        int num_items;
        auto (*get_name)(int idx) -> char const*;
    };
    
#define STR(x) #x
#define BEGIN_REFLECT_IMPL(_cls, _counter)\
    template<int n, typename fake = void>\
    struct _field_info;\
    static constexpr std::string_view _type_name = STR(_cls);\
    static constexpr bool _is_subclass = false;\
    static constexpr int _init_count = _counter;\
    using _my_type = _cls
#define BEGIN_REFLECT_INHERIT_IMPL(_cls, _counter, _parent)\
    template<int n, typename fake = void>\
    struct _field_info;\
    static constexpr std::string_view _type_name = STR(_cls);\
    static constexpr bool _is_subclass = true;\
    static constexpr int _init_count = _counter;\
    using _my_type = _cls;\
    using _parent_type = _parent

#define FIELD_IMPL_MOD(_counter, _var_type, _var_name, _init, ...)\
    _var_type _var_name = _init;\
    template<typename fake> struct _field_info<(_counter) - _init_count - 1, fake> {\
        static constexpr std::string_view var_name = STR(_var_name);\
        static constexpr std::string_view type_name = STR(_var_type);\
        static constexpr auto modifiers = ModifierPack {__VA_ARGS__};\
        static constexpr auto offset = offsetof(_my_type, _var_name);\
        static inline _var_type default_value = _init;\
        using type = _var_type;\
        static constexpr type& get(_my_type& obj) {\
            return obj._var_name;\
        }\
        static constexpr type const& get(_my_type const& obj) {\
            return obj._var_name;\
        }\
        static type const& get_default() {\
            return default_value;\
        }\
        template<typename Modifier>\
        static constexpr auto get_modifier() -> std::optional<Modifier> {\
            std::optional<Modifier> ret = std::nullopt;\
            modifiers.for_each([&ret](auto mod) {\
                if constexpr (std::is_same_v<decltype(mod), Modifier>) {\
                    ret = mod;\
                }\
            });\
            return ret;\
        }\
        using CallbackType = std::function<void(_var_type&, _var_type&, _my_type&)>;\
        static auto register_on_change_callback(CallbackType callback) {\
            _on_change_callbacks.emplace_back(std::move(callback));\
            return _on_change_callbacks.size() - 1;\
        }\
        static void on_change(_var_type old_val, _var_type new_val, _my_type& obj) {\
            for (auto& callback : _on_change_callbacks) {\
                callback(old_val, new_val, obj);\
            }\
        }\
        static void unregister_on_change_callback(int idx) {\
            if (idx < 0 || idx >= _on_change_callbacks.size()) return;\
            _on_change_callbacks.erase(_on_change_callbacks.begin() + idx);\
        }\
        static void unregister_all_on_change_callbacks() {\
            _on_change_callbacks.clear();\
        }\
        static void set_default(type const& val) {\
            default_value = val;\
        }\
private:\
        static inline std::vector<CallbackType> _on_change_callbacks;\
    }
#define END_REFLECT_IMPL(_counter)\
public:\
    template<typename Callable, std::size_t... Is>\
    static constexpr void for_each_field_impl(Callable&& callable, std::index_sequence<Is...>) {\
		if constexpr (sizeof...(Is) == 0) return;\
        (callable(_field_info<Is>{}), ...);\
    }\
    template<typename Callable>\
    static constexpr void for_each_field(Callable&& callable) {\
        for_each_field_impl(std::forward<Callable>(callable), std::make_index_sequence<num_members>{});\
    }\
    static constexpr bool is_reflectable() { return true; }\
    static constexpr auto get_class_name() { return _type_name; }\
    static constexpr int num_members = (_counter) - _init_count - 1;\
    using parent_type = void

#define END_REFLECT_INHERIT_IMPL(_counter)\
public:\
    template<typename Callable, std::size_t... Is>\
    static constexpr void for_each_field_impl(Callable&& callable, std::index_sequence<Is...>) {\
        if constexpr (_is_subclass && _parent_type::is_reflectable())\
            _parent_type::for_each_field(std::forward<Callable>(callable));\
		if constexpr (sizeof...(Is) == 0) return;\
        (callable(_field_info<Is>{}), ...);\
    }\
    template<typename Callable>\
    static constexpr void for_each_field(Callable&& callable) {\
        for_each_field_impl(std::forward<Callable>(callable), std::make_index_sequence<my_num_members>{});\
    }\
    static constexpr bool is_reflectable() { return true; }\
    static constexpr auto get_class_name() { return _type_name; }\
    static constexpr int my_num_members = (_counter) - _init_count - 1;\
    static constexpr int num_members = my_num_members +\
        (_parent_type::is_reflectable() ? _parent_type::num_members : 0);\
    using parent_type = _parent_type

#define BEGIN_REFLECT(cls) BEGIN_REFLECT_IMPL(cls, __COUNTER__)
#define FIELD_MOD(type, var_name, init, ...) FIELD_IMPL_MOD(__COUNTER__, type, var_name, init, __VA_ARGS__)
#define END_REFLECT() END_REFLECT_IMPL(__COUNTER__)

#define BEGIN_REFLECT_INHERIT(cls, parent) BEGIN_REFLECT_INHERIT_IMPL(cls, __COUNTER__, parent)
#define END_REFLECT_INHERIT() END_REFLECT_INHERIT_IMPL(__COUNTER__)

    // type traits
    template<typename T>
    struct is_reflectable {
        template<typename U>
        static auto test(int) -> decltype(U::is_reflectable(), std::true_type{});
        template<typename>
        static auto test(...) -> std::false_type;
        static constexpr bool value = decltype(test<T>(0))::value;
    };

    template<typename T>
    struct has_serialization_callback {
        template<typename U>
        static auto test(int) -> decltype(std::declval<U>().on_serialize(), std::true_type{});
        template<typename>
        static auto test(...) -> std::false_type;
        static constexpr bool value = decltype(test<T>(0))::value;
    };

    template<typename T>
    struct has_deserialization_callback {
        template<typename U>
        static auto test(int) -> decltype(std::declval<U>().on_deserialize(), std::true_type{});
        template<typename>
        static auto test(...) -> std::false_type;
        static constexpr bool value = decltype(test<T>(0))::value;
    };
} // namespace PTS
