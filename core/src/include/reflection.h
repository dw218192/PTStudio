#pragma once
#include <string_view>
#include <utility>
#include <tuple>
#include <optional>
#include <vector>
#include <functional>
#include <memory>
#include <string>

#include "typeTraitsUtil.h"

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
    // used to dynamically get info about a class
    struct ClassInfo {
        std::string_view class_name;
        int num_members;
        ClassInfo const* parent;
    };
    /** @brief non-templated field info, not type-safe \n
     * used to dynamically get info about a field or store a type-erased field pointer
     * @note this is a workaround for the fact that we can't have a virtual template function
    */
    struct FieldInfo {
        friend struct std::hash<FieldInfo>;

        std::string_view var_name;
        std::string_view type_name;
        int index;
        
        bool is_pointer;
        auto operator==(FieldInfo const& other) const noexcept -> bool {
            return var_name == other.var_name && type_name == other.type_name && index == other.index;
        }

        template<typename MemPtr>
        struct Helper {
            MemPtr mem_ptr;

            template<typename = std::enable_if_t<std::is_member_pointer_v<MemPtr>>>
            Helper(MemPtr mem_ptr) : mem_ptr(mem_ptr) {}
        };

        template<typename RetType, typename T>
        auto get(T* obj) const -> RetType& {
            auto mem_ptr = static_cast<Helper<RetType (T::*)>*>(m_any_mem_ptr.get())->mem_ptr;
            return (obj->*mem_ptr);
        }
        template<typename RetType, typename T>
        auto get(T const* obj) const -> RetType const& {
            auto mem_ptr = static_cast<Helper<RetType (T::*)>*>(m_any_mem_ptr.get())->mem_ptr;
            return (obj->*mem_ptr);
        }
        template<template <int, typename> typename TemplatedFieldInfo, int n>
        FieldInfo(TemplatedFieldInfo<n, void> const& info) :
            var_name(info.var_name),
            type_name(info.type_name),
            index(n),
            is_pointer(std::is_pointer_v<typename TemplatedFieldInfo<n, void>::type>),
            m_any_mem_ptr(std::make_shared<Helper<
                decltype(TemplatedFieldInfo<n, void>::mem_pointer)>>
            (info.mem_pointer)) {}
    private:
        std::shared_ptr<void> m_any_mem_ptr;
    };

#define STR(x) #x
#define BEGIN_REFLECT_IMPL(_cls, _counter, _parent)\
    using _my_type = _cls;\
    using _parent_type = _parent;\
    template<int n, typename fake = void>\
    struct _field_info;\
    static constexpr std::string_view _type_name = STR(_cls);\
    static constexpr bool _is_subclass = Traits::is_reflectable<_parent_type>::value;\
    static constexpr int _init_count = _counter

#define FIELD_IMPL(_counter, _var_type, _var_name, _init, ...)\
    _var_type _var_name = _init;\
    template<typename fake> struct _field_info<(_counter) - _init_count - 1, fake> {\
        static constexpr std::string_view var_name = STR(_var_name);\
        static constexpr std::string_view type_name = STR(_var_type);\
        static constexpr auto modifiers = ModifierPack {__VA_ARGS__};\
        static constexpr auto mem_pointer = &_my_type::_var_name;\
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
    template<typename Callable, typename U, std::size_t... Is>\
    static constexpr\
    std::enable_if_t<!std::is_same_v<U, void>>\
    for_each_field_impl(Callable&& callable, std::index_sequence<Is...>) {\
        U::for_each_field(std::forward<Callable>(callable));\
		if constexpr (sizeof...(Is) > 0)\
            (callable(_field_info<Is>{}), ...);\
    }\
    template<typename Callable, typename U, std::size_t... Is>\
    static constexpr\
    std::enable_if_t<std::is_same_v<U, void>>\
    for_each_field_impl(Callable&& callable, std::index_sequence<Is...>) {\
        if constexpr (sizeof...(Is) > 0)\
            (callable(_field_info<Is>{}), ...);\
    }\
    template<typename Callable>\
    static constexpr void for_each_field(Callable&& callable) {\
        for_each_field_impl<Callable, _parent_type>\
            (std::forward<Callable>(callable), std::make_index_sequence<my_num_members>{});\
    }\
    static constexpr std::string_view const& get_class_name() { return _type_name; }\
    static constexpr int my_num_members = (_counter) - _init_count - 1;\
    static constexpr int num_members = my_num_members + Traits::get_num_members<_parent_type>::value

#define DECL_DYNAMIC_INFO()\
public:\
    static constexpr ClassInfo _class_info = {\
        _type_name,\
        num_members,\
        Traits::get_class_info<_parent_type>::value\
    };\
    virtual ClassInfo const& get_class_info() const {\
        return _class_info;\
    }\
    virtual std::vector<FieldInfo> get_field_info() const {\
        std::vector<FieldInfo> ret;\
        _my_type::for_each_field([&ret](auto field) { ret.emplace_back(field); });\
        return ret;\
    }\
    template<typename T>\
    std::enable_if_t<Traits::has_dynamic_info<T>::value && std::is_base_of_v<_my_type, T>, T*>\
    as() {\
        if (get_class_info().class_name == T::get_class_name()) {\
            return dynamic_cast<T*>(this);\
        }\
        return nullptr;\
    }\
    template<typename T>\
    std::enable_if_t<Traits::has_dynamic_info<T>::value && std::is_base_of_v<_my_type, T>, T const*>\
    as() const {\
        if (get_class_info().class_name == T::get_class_name()) {\
            return dynamic_cast<T const*>(this);\
        }\
        return nullptr;\
    }
#define BEGIN_REFLECT(cls, parent) BEGIN_REFLECT_IMPL(cls, __COUNTER__, parent)
#define FIELD(type, var_name, init, ...) FIELD_IMPL(__COUNTER__, type, var_name, init, __VA_ARGS__)
#define END_REFLECT() END_REFLECT_IMPL(__COUNTER__)

    template<typename Reflected>
    std::enable_if_t<Traits::is_reflectable<Reflected>::value && Traits::has_dynamic_info<Reflected>::value>
    dynamic_for_each_field(Reflected& obj, std::function<void(FieldInfo const&)> callable) {
        for (auto& info : obj.get_field_info()) {
            callable(info);
        }
    }
} // namespace PTS


// provide hash functions for std::unordered_map
namespace std {
    template<>
    struct hash<PTS::ClassInfo> {
        auto operator()(PTS::ClassInfo const& info) const noexcept -> size_t {
            return hash<std::string_view>{}(info.class_name);
        }
    };
    template<>
    struct hash<PTS::FieldInfo> {
        auto operator()(PTS::FieldInfo const& info) const noexcept -> size_t {
            return hash<std::string>{}(std::string{ info.type_name } + " " + 
                std::string{ info.var_name } + " " + std::to_string(info.index));
        }
    };
}; // namespace std