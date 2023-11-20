#pragma once
#include <string_view>
#include <utility>
#include <tuple>
#include <optional>
namespace PTS {
    /**
     * simple reflection framework
     * limitations:
     *  - doesn't include inherited members
     *  - only single declaration per line
     *  - FIELD(type<a,b,c,..>, name) doesn't work, alias the type first
     *  - relies on non-standard __COUNTER__ macro, but g++, clang, and msvc all support it
     *  - access control change in END_REFLECT() may cause bugs
    */

    /**
     * a modifier is an attribute that can be attached to a field
     * modifiers are used to change the behavior of the inspector
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
    template<typename T>
    struct MMin {
        T value;
        static constexpr std::string_view name = "min";
        constexpr MMin(T val) : value(val) {}
    };
    template<typename T>
    struct MMax {
        T value;
        static constexpr std::string_view name = "max";
        constexpr MMax(T val) : value(val) {}
    };
    template<typename T>
    struct MRange {
        T min, max, step;
        static constexpr std::string_view name = "range";
        constexpr MRange(T min, T max, T step) : min(min), max(max), step(step) {}
    };
    struct MSerialize {
        static constexpr std::string_view name = "serialize";
    };
    struct MNoInspect {
        static constexpr std::string_view name = "no inspect";
    };
    struct MColor {
        static constexpr std::string_view name = "color";
    };

#define STR(x) #x
#define BEGIN_REFLECT_IMPL(_cls, _counter)\
    template<int n, typename fake = void>\
    struct _field_info;\
    static constexpr std::string_view _type_name = STR(_cls);\
    static constexpr int _init_count = _counter;\
    using _my_type = _cls

#define FIELD_IMPL(_counter, _var_type, _var_name)\
    _var_type _var_name;\
    template<typename fake> struct _field_info<(_counter) - _init_count - 1, fake> {\
        static constexpr std::string_view var_name = STR(_var_name);\
        static constexpr std::string_view type_name = STR(_var_type);\
        static constexpr auto offset = offsetof(_my_type, _var_name);\
        using type = _var_type;\
        static constexpr type& get(_my_type& obj) {\
            return obj._var_name;\
        }\
        static constexpr type const& get(_my_type const& obj) {\
            return obj._var_name;\
        }\
        template<typename Modifier>\
        static constexpr auto get_modifier() -> std::optional<Modifier> { return std::nullopt; }\
    }
#define FIELD_IMPL_MOD(_counter, _var_type, _var_name, _init, ...)\
    _var_type _var_name = _init;\
    template<typename fake> struct _field_info<(_counter) - _init_count - 1, fake> {\
        static constexpr std::string_view var_name = STR(_var_name);\
        static constexpr std::string_view type_name = STR(_var_type);\
        static constexpr auto modifiers = ModifierPack {__VA_ARGS__};\
        static constexpr auto offset = offsetof(_my_type, _var_name);\
        using type = _var_type;\
        static constexpr type& get(_my_type& obj) {\
            return obj._var_name;\
        }\
        static constexpr type const& get(_my_type const& obj) {\
            return obj._var_name;\
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
    static constexpr int num_members = (_counter) - _init_count - 1

#define BEGIN_REFLECT(cls) BEGIN_REFLECT_IMPL(cls, __COUNTER__)
#define FIELD(type, var_name) FIELD_IMPL(__COUNTER__, type, var_name)
#define FIELD_MOD(type, var_name, init, ...) FIELD_IMPL_MOD(__COUNTER__, type, var_name, init, __VA_ARGS__)
#define END_REFLECT() END_REFLECT_IMPL(__COUNTER__)

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
}