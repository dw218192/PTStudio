#pragma once
#include <string_view>

/**
 * simple reflection
 * limitations:
 *  - doesn't include inherited members
 *  - only single declaration per line
 *  - FIELD(type<a,b,c,..>, name) doesn't work, alias the type first
 *  - relies on non-standard __COUNTER__ macro, but g++, clang, and msvc all support it
 *  - access control change in END_REFLECT() may cause bugs
*/
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
        static constexpr std::size_t offset = offsetof(_my_type, _var_name);\
        using type = _var_type;\
        type& get(_my_type& obj) {\
            return obj._var_name;\
        }\
        type const& get(_my_type const& obj) {\
            return obj._var_name;\
        }\
    }
#define FIELD_IMPL_INIT(_counter, _var_type, _var_name, _val)\
    _var_type _var_name { _val };\
    template<typename fake> struct _field_info<(_counter) - _init_count - 1, fake> {\
        static constexpr std::string_view var_name = STR(_var_name);\
        static constexpr std::string_view type_name = STR(_var_type);\
        static constexpr std::size_t offset = offsetof(_my_type, _var_name);\
        using type = _var_type;\
        type& get(_my_type& obj) {\
            return obj._var_name;\
        }\
        type const& get(_my_type const& obj) {\
            return obj._var_name;\
        }\
    }
#define END_REFLECT_IMPL(_counter)\
public:\
    template<typename Callable, std::size_t... Is>\
    static void for_each_field_impl(Callable&& callable, std::index_sequence<Is...>) {\
		if constexpr (sizeof...(Is) == 0) return;\
        (callable(_field_info<Is>{}), ...);\
    }\
    template<typename Callable>\
    static void for_each_field(Callable&& callable) {\
        for_each_field_impl(std::forward<Callable>(callable), std::make_index_sequence<num_members>{});\
    }\
    static constexpr bool is_reflectable() { return true; }\
    static constexpr auto type_name() { return _type_name; }\
    static constexpr int num_members = (_counter) - _init_count - 1

#define BEGIN_REFLECT(cls) BEGIN_REFLECT_IMPL(cls, __COUNTER__)
#define FIELD(type, var_name) FIELD_IMPL(__COUNTER__, type, var_name)
#define FIELD_INIT(type, var_name, val) FIELD_IMPL_INIT(__COUNTER__, type, var_name, val)
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