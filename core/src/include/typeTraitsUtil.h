#pragma once
#include <type_traits>
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
    } // namespace Traits
} // namespace PTS