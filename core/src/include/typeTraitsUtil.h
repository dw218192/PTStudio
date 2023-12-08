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
            static auto test(int) -> decltype(std::declval<U>().get_class_info(), std::true_type{});
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

        /**
         * @brief Removes all cv qualifiers from a pointer type
         * @tparam T The type to remove cv qualifiers from
         * @return The type with all cv qualifiers removed, e.g. `int const volatile* const*` becomes `int**`
        */
        template<typename T>
        struct extract_raw_type {
            template<typename U, typename...>
            static constexpr auto get_type(U*, std::enable_if_t<!is_container<U>::value>* = nullptr) {
                using decayed = std::decay_t<U>;
                if constexpr (std::is_pointer_v<decayed>) {
                    return std::add_pointer_t<decltype(get_type(decayed{}))>{};
                } else {
                    return decayed {};
                }
            }
            template<template<typename...> typename U, typename... Ts>
            static constexpr auto get_type(U<Ts...>*) {
                return U<decltype(get_type(std::add_pointer_t<Ts>{}))...>{};
            }
            using type = decltype(get_type(std::add_pointer_t<T>{}));
        };

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