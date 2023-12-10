#pragma once
#include <type_traits>
#include <iterator>
#include <tuple>
#include <utility>
#include <string_view>

#include "utils.h"
namespace PTS {
    struct ClassInfo;

    namespace Traits {
        template<typename T>
        struct is_reflectable {
            template<typename U>
            static auto test(int) -> decltype(U::num_members, std::true_type{});
            template<typename>
            static auto test(...) -> std::false_type;
            static constexpr bool value = decltype(test<T>(0))::value;
        };

        template<typename T>
        struct get_num_members {
            template<typename U, typename = std::void_t<decltype(U::num_members)>>
            static constexpr auto get(int) -> int { return U::num_members; }
            template<typename, typename = void>
            static constexpr auto get(...) -> int { return 0; }
            static constexpr auto value = get<T>(0);
        };

        template<typename T>
        struct get_class_info {
            template<typename U, typename = std::void_t<decltype(U::_class_info)>>
            static constexpr auto get(int) -> ClassInfo const* { return &U::_class_info; }
            template<typename, typename = void>
            static constexpr auto get(...) -> ClassInfo const* { return nullptr; }
            static constexpr auto value = get<T>(0);
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

        template<typename T>
        struct has_dynamic_info {
            template<typename U>
            static auto test(int) -> decltype(std::declval<U>().dyn_get_class_info(), std::true_type{});
            template<typename>
            static auto test(...) -> std::false_type;
            static constexpr bool value = decltype(test<T>(0))::value;
        };

        template<typename T>
        struct is_container {
            template<typename U>
            static auto test(int) -> decltype(std::declval<U>().begin(), std::declval<U>().end(), std::true_type{});
            template<typename>
            static auto test(...) -> std::false_type;
            static constexpr bool value = decltype(test<T>(0))::value;
        };

        template<typename T>
        struct is_pair {
            template<typename U>
            static auto test(int) -> decltype(std::declval<U>().first, std::declval<U>().second, std::true_type{});
            template<typename>
            static auto test(...) -> std::false_type;
            static constexpr bool value = decltype(test<T>(0))::value;
        };

        template<typename T>
        struct is_tuple {
            template<typename U>
            static auto test(int) -> decltype(std::tuple_size<U>::value, std::true_type{});
            template<typename>
            static auto test(...) -> std::false_type;
            static constexpr bool value = decltype(test<T>(0))::value;
        };
        
        template <typename T>
        struct is_iterator {
            template <typename U,
                typename=typename std::iterator_traits<U>::difference_type,
                typename=typename std::iterator_traits<U>::pointer,
                typename=typename std::iterator_traits<U>::reference,
                typename=typename std::iterator_traits<U>::value_type,
                typename=typename std::iterator_traits<U>::iterator_category
            >
            static auto test(int) -> std::true_type;
            template <typename>
            static auto test(...) -> std::false_type;
            static constexpr bool value = decltype(test<T>(0))::value;
        };

        template<typename T, typename...>
        struct extract_raw_type {
            using type = std::decay_t<T>;
        };
        template<typename T>
        struct extract_raw_type<T*> {
            using type = typename extract_raw_type<std::decay_t<T>>::type*;
        };
        template<typename T>
        struct extract_raw_type<T* const> {
            using type = typename extract_raw_type<std::decay_t<T>>::type*;
        };
        template<typename T>
        struct extract_raw_type<T* volatile> {
            using type = typename extract_raw_type<std::decay_t<T>>::type*;
        };
        template<typename T>
        struct extract_raw_type<T* const volatile> {
            using type = typename extract_raw_type<std::decay_t<T>>::type*;
        };
        template<template<typename...> typename T, typename... Args>
        struct extract_raw_type<T<Args...>> {
            using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
        };
        template<template<typename...> typename T, typename... Args>
        struct extract_raw_type<T<Args...> const> {
            using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
        };
        template<template<typename...> typename T, typename... Args>
        struct extract_raw_type<T<Args...> volatile> {
            using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
        };
        template<template<typename...> typename T, typename... Args>
        struct extract_raw_type<T<Args...> const volatile> {
            using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
        };

        /**
         * @brief Removes all cv qualifiers from a pointer type
         * @tparam T The type to remove cv qualifiers from
         * @return The type with all cv qualifiers removed, e.g. `int const volatile* const*` becomes `int**`
        */
        template<typename T>
        using extract_raw_type_t = typename extract_raw_type<T>::type;
        
        template<typename Arg, typename... Args>
        struct contains {
            static constexpr bool value = (std::is_same_v<Arg, Args> || ...);
        };
        template<typename Arg, typename... Args>
        constexpr bool contains_v = contains<Arg, Args...>::value;

    } // namespace Traits
} // namespace PTS