#pragma once
#include <string_view>
#include <utility>
#include <tuple>
#include <optional>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <cstddef>
#include <typeinfo>

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

        // not marked as constexpr because void* conversion is prohibited in constexpr
        template<typename Mod>
        auto get() const -> Mod const* {
            void const* ret = nullptr;
            std::apply([&](auto&&... args) {
                ([&](auto&& arg) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, Mod>) {
                        ret = &arg;
                    }
                }(args), ...);
            }, args);
            return static_cast<Mod const*>(ret);
        }
        template<typename Mod>
        constexpr auto has() const {
            return (std::is_same_v<Modifiers, Mod> || ...);
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

    // interface for type-erased iterators
    struct TypeErasedIteratorInterface {
        virtual ~TypeErasedIteratorInterface() = default;
        virtual auto operator++() -> TypeErasedIteratorInterface& = 0;
        virtual auto operator*() const -> void const* = 0;
        virtual auto operator*() -> void* = 0;
        virtual auto operator==(TypeErasedIteratorInterface const& other) const -> bool = 0;
        virtual auto operator!=(TypeErasedIteratorInterface const& other) const -> bool = 0;
    };
    // type-erased iterator
    struct TypeErasedIterator {
        template<typename Iterator, typename = std::enable_if_t<Traits::is_iterator<Iterator>::value>>
        struct IteratorWrapper : TypeErasedIteratorInterface {
            IteratorWrapper(Iterator it) : m_it(it) {}
            auto operator++() -> TypeErasedIteratorInterface& override {
                ++m_it;
                return *this;
            }
            auto operator*() const -> void const* override {
                return &(*m_it);
            }
            auto operator*() -> void* override {
                return &(*m_it);
            }
            auto operator==(TypeErasedIteratorInterface const& other) const -> bool override {
                if (auto* other_it = dynamic_cast<IteratorWrapper const*>(&other)) {
                    return m_it == other_it->m_it;
                }
                return false;
            }
            auto operator!=(TypeErasedIteratorInterface const& other) const -> bool override {
                return !(*this == other);
            }
        private:
            Iterator m_it;
        };
        
        template<typename Iterator, typename = std::enable_if_t<Traits::is_iterator<Iterator>::value>>
        TypeErasedIterator(Iterator it) : m_impl(std::make_shared<IteratorWrapper<Iterator>>(it)) {}
        template<typename T>
        TypeErasedIterator(T* ptr) : m_impl(std::make_shared<IteratorWrapper<T*>>(ptr)) {}
        TypeErasedIterator() : m_impl(nullptr) {}

        auto operator++() -> TypeErasedIteratorInterface& {
            return m_impl->operator++();
        }
        auto operator*() const -> void const* {
            return m_impl->operator*();
        }
        auto operator*() -> void* {
            return m_impl->operator*();
        }
        auto operator==(TypeErasedIterator const& other) const -> bool {
            return m_impl->operator==(*other.m_impl);
        }
        auto operator!=(TypeErasedIterator const& other) const -> bool {
            return m_impl->operator!=(*other.m_impl);
        }
    private:
        std::shared_ptr<TypeErasedIteratorInterface> m_impl;
    };

    /**
     * @brief represents a type, used to dynamically get info about a field
    */
    struct Type {
        std::size_t normalized_type_hash {0}; // no good way to get a readable name as of C++17
        bool is_pointer {false};
        bool is_container {false};
        bool is_reflectable {false};
        bool is_arithmetic {false};

        template<typename T>
        static auto of() -> Type {
            auto ret = Type{};
            using raw_t = Traits::extract_raw_type_t<T>;
            ret.normalized_type_hash = typeid(raw_t).hash_code();
            if constexpr (std::is_pointer_v<raw_t>) {
                ret.is_pointer = true;
                ret.m_pointed_to_type = std::make_shared<Type>(of<std::remove_pointer_t<raw_t>>());
            }
            if constexpr (Traits::is_container<raw_t>::value) {
                ret.is_container = true;
                ret.m_contained_type = std::make_shared<Type>(of<typename raw_t::value_type>());
            }
            if constexpr (Traits::is_reflectable<raw_t>::value) {
                ret.is_reflectable = true;
                if constexpr (!std::is_same_v<typename raw_t::parent_type, void>) {
                    ret.m_inherited_from_type = std::make_shared<Type>(of<typename raw_t::parent_type>());
                }
            }
            if constexpr (std::is_arithmetic_v<raw_t>) {
                ret.is_arithmetic = true;
            }
            return ret;
        }
        
        Type(Type const&) = default;
        Type(Type&&) = default;
        Type& operator=(Type const&) = default;
        Type& operator=(Type&&) = default;
        ~Type() = default;

        auto is_valid() const noexcept {
            return normalized_type_hash != 0;
        }
        auto pointed_to_type() const -> Type {
            return m_pointed_to_type ? *m_pointed_to_type : Type{};
        }
        auto inherited_from_type() const -> Type {
            return m_inherited_from_type ? *m_inherited_from_type : Type{};
        }
        auto contained_type() const -> Type {
            return m_contained_type ? *m_contained_type : Type{};
        }
        auto has_common_base_with(Type const& other) const -> bool {
            if (!is_valid() || !other.is_valid()) return false;
            if (normalized_type_hash == other.normalized_type_hash) {
                return true;
            } else if (m_inherited_from_type) {
                return m_inherited_from_type->has_common_base_with(other) ||
                    has_common_base_with(other.inherited_from_type());
            } else {
                return false;
            }
        }
        auto operator==(Type const& other) const noexcept -> bool {
            return normalized_type_hash == other.normalized_type_hash;
        }
        auto operator<(Type const& other) const noexcept -> bool {
            return normalized_type_hash < other.normalized_type_hash;
        }
        auto operator!=(Type const& other) const noexcept -> bool {
            return !(*this == other);
        }
        auto operator<=(Type const& other) const noexcept -> bool {
            return *this < other || *this == other;
        }
        auto operator>(Type const& other) const noexcept -> bool {
            return !(*this <= other);
        }
        auto operator>=(Type const& other) const noexcept -> bool {
            return !(*this < other);
        }
    private:
        Type() = default;
        std::shared_ptr<Type> m_pointed_to_type {};
        std::shared_ptr<Type> m_inherited_from_type {};
        std::shared_ptr<Type> m_contained_type {};
    };

    /** @brief non-templated field info, not type-safe \n
     * used to dynamically get info about a field or store a type-erased field pointer
     * @note this is a workaround for the fact that we can't have a virtual template function
    */
    struct FieldInfo {
        friend struct std::hash<FieldInfo>;

        std::string_view var_name;
        std::string_view type_name;
        Type type;

        // indicates how many fields come before this one in the class
        // does not account for fields in parent classes
        int index;

        auto operator==(FieldInfo const& other) const noexcept -> bool {
            return var_name == other.var_name && type_name == other.type_name && index == other.index;
        }
        auto operator!=(FieldInfo const& other) const noexcept -> bool {
            return !(*this == other);
        }
        auto operator<(FieldInfo const& other) const noexcept -> bool {
            return std::tuple{ var_name, type_name, index } < std::tuple{ other.var_name, other.type_name, other.index };
        }
        auto operator<=(FieldInfo const& other) const noexcept -> bool {
            return *this < other || *this == other;
        }
        auto operator>(FieldInfo const& other) const noexcept -> bool {
            return !(*this <= other);
        }
        auto operator>=(FieldInfo const& other) const noexcept -> bool {
            return !(*this < other);
        }
        auto begin() const -> TypeErasedIterator {
            return m_begin;
        }
        auto end() const -> TypeErasedIterator {
            return m_end;
        }

        template<template <int, typename> typename TemplatedFieldInfo, int n, typename Reflected, 
            typename = std::enable_if_t<Traits::is_reflectable<Reflected>::value>>
        FieldInfo(TemplatedFieldInfo<n, void> const& info, Reflected& obj) :
            var_name { info.var_name },
            type_name { info.type_name },
            type { Type::of<typename TemplatedFieldInfo<n, void>::type>() },
            index { n }            
        {
            using FieldType = typename TemplatedFieldInfo<n, void>::type;
            if constexpr (Traits::is_container<FieldType>::value) {
                using Iter = typename FieldType::iterator;
                auto& container = const_cast<FieldType&>(obj.*info.mem_pointer);
                m_begin = TypeErasedIterator { container.begin() };
                m_end = TypeErasedIterator { container.end() };
            } else {
                auto& member = const_cast<FieldType&>(obj.*info.mem_pointer);
                m_begin = TypeErasedIterator { std::addressof(member) };
                m_end = TypeErasedIterator { std::addressof(member) + 1 };
            }
        }
    private:
        TypeErasedIterator m_begin, m_end; // type-erased iterators if the field is a container
    };

#define STR(x) #x
#define BEGIN_REFLECT_IMPL(_cls, _counter, _parent)\
    using _my_type = _cls;\
    using _parent_type = _parent;\
    template<int n, typename fake = void>\
    struct _field_info;\
    static constexpr std::string_view _type_name = STR(_cls);\
    static constexpr bool _is_subclass = Traits::is_reflectable<_parent_type>::value;\
    static constexpr int _init_count = _counter;\

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
        static auto get_modifier() {\
            return modifiers.template get<Modifier>();\
        }\
        template<typename Modifier>\
        static constexpr bool has_modifier() {\
            return modifiers.template has<Modifier>();\
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
    static constexpr int num_members = my_num_members + Traits::get_num_members<_parent_type>::value;\
    using parent_type = _parent_type

#define DECL_DYNAMIC_INFO()\
public:\
    static constexpr ClassInfo _class_info = {\
        _type_name,\
        num_members,\
        Traits::get_class_info<_parent_type>::value\
    };\
    virtual ClassInfo const& dyn_get_class_info() const {\
        return _class_info;\
    }\
    virtual std::vector<FieldInfo> dyn_get_field_infos() const {\
        std::vector<FieldInfo> ret;\
        _my_type::for_each_field([&ret, this](auto field) { ret.emplace_back(field, *this); });\
        return ret;\
    }\
    template<typename T>\
    std::enable_if_t<Traits::has_dynamic_info<T>::value && std::is_base_of_v<_my_type, T>, T*>\
    as() {\
        if (dyn_get_class_info().class_name == T::get_class_name()) {\
            return dynamic_cast<T*>(this);\
        }\
        return nullptr;\
    }\
    template<typename T>\
    std::enable_if_t<Traits::has_dynamic_info<T>::value && std::is_base_of_v<_my_type, T>, T const*>\
    as() const {\
        if (dyn_get_class_info().class_name == T::get_class_name()) {\
            return dynamic_cast<T const*>(this);\
        }\
        return nullptr;\
    }\
    void dyn_for_each_field(std::function<void(FieldInfo const&)> callable) {\
        for (auto& info : dyn_get_field_infos()) {\
            callable(info);\
        }\
    }

#define BEGIN_REFLECT(cls, parent) BEGIN_REFLECT_IMPL(cls, __COUNTER__, parent)
#define FIELD(type, var_name, init, ...) FIELD_IMPL(__COUNTER__, type, var_name, init, __VA_ARGS__)
#define END_REFLECT() END_REFLECT_IMPL(__COUNTER__)
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
    template<>
    struct hash<PTS::TypeErasedIterator> {
        auto operator()(PTS::TypeErasedIterator const& it) const noexcept -> size_t {
            return hash<std::uintptr_t>{}(reinterpret_cast<std::uintptr_t>(*it));
        }
    };
    template<>
    struct hash<PTS::Type> {
        auto operator()(PTS::Type const& type) const noexcept -> size_t {
            return type.normalized_type_hash;
        }
    };
}; // namespace std