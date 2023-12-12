#pragma once
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace PTS {
struct ClassInfo;

namespace Traits {
template <typename T, typename = void>
struct is_reflectable : std::false_type {};
template <typename T>
struct is_reflectable<T,
                      std::void_t<decltype(std::declval<T>().get_class_name())>>
    : std::true_type {};
template <typename T>
constexpr bool is_reflectable_v = is_reflectable<T>::value;

template <typename T, typename = void>
struct get_num_members : std::integral_constant<int, 0> {};
template <typename T>
struct get_num_members<T, std::void_t<decltype(T::num_members)>>
    : std::integral_constant<int, T::num_members> {};
template <typename T>
constexpr auto get_num_members_v = get_num_members<T>::value;

template <typename T>
struct get_class_info {
    template <typename U, typename = std::void_t<decltype(U::_class_info)>>
    static constexpr auto get(int) -> ClassInfo const* {
        return &U::_class_info;
    }
    template <typename, typename = void>
    static constexpr auto get(...) -> ClassInfo const* {
        return nullptr;
    }
    static constexpr auto value = get<T>(0);
};
template <typename T>
constexpr auto get_class_info_v = get_class_info<T>::value;

template <typename T, typename = void>
struct has_serialization_callback : std::false_type {};
template <typename T>
struct has_serialization_callback<
    T,
    std::void_t<decltype(std::declval<T>().on_serialize())>> : std::true_type {
};
template <typename T>
constexpr bool has_serialization_callback_v =
    has_serialization_callback<T>::value;

template <typename T, typename = void>
struct has_deserialization_callback : std::false_type {};
template <typename T>
struct has_deserialization_callback<
    T,
    std::void_t<decltype(std::declval<T>().on_deserialize())>>
    : std::true_type {};
template <typename T>
constexpr bool has_deserialization_callback_v =
    has_deserialization_callback<T>::value;

template <typename T, typename = void>
struct has_dynamic_info : std::false_type {};
template <typename T>
struct has_dynamic_info<
    T,
    std::void_t<decltype(std::declval<T>().dyn_get_class_info())>>
    : std::true_type {};
template <typename T>
constexpr bool has_dynamic_info_v = has_dynamic_info<T>::value;

template <typename T, typename = void>
struct is_container : std::false_type {};
template <typename T>
struct is_container<T,
                    std::void_t<decltype(std::declval<T>().begin()),
                                decltype(std::declval<T>().end())>>
    : std::true_type {};
template <typename T>
constexpr bool is_container_v = is_container<T>::value;

template <typename T>
struct is_pair : std::false_type {};
template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};
template <typename T>
constexpr auto is_pair_v = is_pair<T>::value;

template <typename T>
struct is_tuple : std::false_type {};
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};
template <typename T>
constexpr auto is_tuple_v = is_tuple<T>::value;

template <typename T>
struct is_tuple_like : std::bool_constant<is_tuple_v<T> || is_pair_v<T>> {};
template <typename T>
constexpr auto is_tuple_like_v = is_tuple_like<T>::value;

template <typename T, typename = void>
struct is_iterator : std::false_type {};
template <typename T>
struct is_iterator<
    T,
    std::void_t<typename std::iterator_traits<T>::difference_type,
                typename std::iterator_traits<T>::pointer,
                typename std::iterator_traits<T>::reference,
                typename std::iterator_traits<T>::value_type,
                typename std::iterator_traits<T>::iterator_category>>
    : std::true_type {};
template <typename T>
constexpr auto is_iterator_v = is_iterator<T>::value;

template <typename T, typename...>
struct extract_raw_type {
    using type = std::decay_t<T>;
};
template <typename T>
struct extract_raw_type<T*> {
    using type = typename extract_raw_type<std::decay_t<T>>::type*;
};
template <typename T>
struct extract_raw_type<T* const> {
    using type = typename extract_raw_type<std::decay_t<T>>::type*;
};
template <typename T>
struct extract_raw_type<T* volatile> {
    using type = typename extract_raw_type<std::decay_t<T>>::type*;
};
template <typename T>
struct extract_raw_type<T* const volatile> {
    using type = typename extract_raw_type<std::decay_t<T>>::type*;
};
template <template <typename...> typename T, typename... Args>
struct extract_raw_type<T<Args...>> {
    using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
};
template <template <typename...> typename T, typename... Args>
struct extract_raw_type<T<Args...> const> {
    using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
};
template <template <typename...> typename T, typename... Args>
struct extract_raw_type<T<Args...> volatile> {
    using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
};
template <template <typename...> typename T, typename... Args>
struct extract_raw_type<T<Args...> const volatile> {
    using type = std::decay_t<T<typename extract_raw_type<Args>::type...>>;
};

/**
 * @brief Removes all cv qualifiers from a pointer type
 * @tparam T The type to remove cv qualifiers from
 * @return The type with all cv qualifiers removed, e.g. `int const volatile*
 * const*` becomes `int**`
 */
template <typename T>
using extract_raw_type_t = typename extract_raw_type<T>::type;

// type searching utils
template <typename Haystack, typename Needle>
struct find;

template <template <typename...> typename Haystack,
          typename Needle,
          typename... Args>
struct find<Haystack<Args...>, Needle>
    : std::integral_constant<std::size_t, static_cast<std::size_t>(-1)> {
    template <std::size_t I>
    static constexpr auto impl() {
        if constexpr (I == sizeof...(Args)) {
            return static_cast<std::size_t>(-1);
        } else if constexpr (std::is_same_v<
                                 Needle,
                                 std::tuple_element_t<I, Haystack<Args...>>>) {
            return I;
        } else {
            return impl<I + 1>();
        }
    }
    static constexpr auto value = impl<0>();
};

template <typename Haystack, typename Needle>
constexpr auto find_v = find<Haystack, Needle>::value;

template <typename Haystack, typename Needle>
struct contains : std::bool_constant<find<Haystack, Needle>::value !=
                                     static_cast<std::size_t>(-1)> {};
template <typename Haystack, typename Needle>
constexpr bool contains_v = contains<Haystack, Needle>::value;

template <typename TupleLike, typename Callable>
constexpr auto for_each(TupleLike&& tuple, Callable&& callable) {
    if constexpr (is_tuple_like_v<std::decay_t<TupleLike>>) {
        std::apply([&callable](auto&&... args) { (callable(std::forward(args)), ...); },
                   std::forward<TupleLike>(tuple));
    } else {
        static_assert(is_tuple_like_v<std::decay_t<TupleLike>>,
                      "for_each requires a tuple-like type");
    }
}
}  // namespace Traits
}  // namespace PTS