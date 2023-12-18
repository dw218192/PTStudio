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

template <typename T, typename = void>
struct is_associative_container : std::false_type {};
template <typename T>
struct is_associative_container<
    T,
    std::void_t<decltype(std::declval<T>().find(std::declval<typename T::key_type>())),
                decltype(std::declval<T>().count(std::declval<typename T::key_type>()))>>
    : std::enable_if_t<is_container_v<T>, std::true_type> {};
template <typename T>
constexpr bool is_associative_container_v = is_associative_container<T>::value;

template <typename T, typename = void>
struct is_sequence_container : std::false_type {};
template <typename T>
struct is_sequence_container<
    T,
    std::void_t<decltype(std::declval<T>().size()),
                decltype(std::declval<T>().empty()),
                decltype(std::declval<T>().rbegin()),
                decltype(std::declval<T>().rend()),
                decltype(std::declval<T>().cbegin()),
                decltype(std::declval<T>().cend()),
                decltype(std::declval<T>().crbegin()),
                decltype(std::declval<T>().crend()),
                decltype(std::declval<T>().front()),
                decltype(std::declval<T>().back()),
                decltype(std::declval<T>().push_back(std::declval<typename T::value_type>())),
                decltype(std::declval<T>().pop_back()),
                decltype(std::declval<T>().resize(std::declval<typename T::size_type>())),
                decltype(std::declval<T>().resize(std::declval<typename T::size_type>(),
                                                   std::declval<typename T::value_type>()))>>
    : std::enable_if_t<is_container_v<T>, std::true_type> {};

template <typename T>
constexpr bool is_sequence_container_v = is_sequence_container<T>::value;

template <typename L, typename R, typename = void>
struct is_equitable : std::false_type {};

template <typename L, typename R>
struct is_equitable<L, R, std::void_t<decltype(std::declval<L>() == std::declval<R>())>>
    : std::true_type {};

template <typename L, typename R>
constexpr bool is_equitable_v = is_equitable<L, R>::value;

template <typename L, typename R, typename = void>
struct is_less_than_comparable : std::false_type {};

template <typename L, typename R>
struct is_less_than_comparable<L, R, std::void_t<decltype(std::declval<L>() < std::declval<R>())>>
    : std::true_type {};

template <typename L, typename R>
constexpr bool is_less_than_comparable_v = is_less_than_comparable<L, R>::value;

template <typename L, typename R, typename = void>
struct is_greater_than_comparable : std::false_type {};

template <typename L, typename R>
struct is_greater_than_comparable<L, R, std::void_t<decltype(std::declval<L>() > std::declval<R>())>>
    : std::true_type {};

template <typename L, typename R>
constexpr bool is_greater_than_comparable_v = is_greater_than_comparable<L, R>::value;

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
// for references, lvalue references are treated as pointers
// rvalue references are treated as values
template <typename T>
struct extract_raw_type<T&> {
    using type = typename extract_raw_type<T>::type*;
};
template <typename T>
struct extract_raw_type<T&&> {
    using type = typename extract_raw_type<T>::type;
};

// templated types
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

// function pointers
template <typename R, typename... Args>
struct extract_raw_type<R(*)(Args...)> {
    using type = typename extract_raw_type<R>::type (*)(typename extract_raw_type<Args>::type...);
};
template <typename R, typename... Args>
struct extract_raw_type<R(* const)(Args...)> {
    using type = typename extract_raw_type<R>::type (*)(typename extract_raw_type<Args>::type...);
};
template <typename R, typename... Args>
struct extract_raw_type<R(* volatile)(Args...)> {
    using type = typename extract_raw_type<R>::type (*)(typename extract_raw_type<Args>::type...);
};
template <typename R, typename... Args>
struct extract_raw_type<R(* const volatile)(Args...)> {
    using type = typename extract_raw_type<R>::type (*)(typename extract_raw_type<Args>::type...);
};
template <typename R, typename... Args>
struct extract_raw_type<R(&)(Args...)> {
    using type = typename extract_raw_type<R(*)(Args...)>::type;
};
template <typename R, typename... Args>
struct extract_raw_type<R(&&)(Args...)> {
    using type = typename extract_raw_type<R(Args...)>::type;
};

// array pointers
template <typename T, std::size_t N>
struct extract_raw_type<T(*)[N]> {
    using type = typename extract_raw_type<T>::type(*)[N];
};
template <typename T, std::size_t N>
struct extract_raw_type<T(* const)[N]> {
    using type = typename extract_raw_type<T>::type(*)[N];
};
template <typename T, std::size_t N>
struct extract_raw_type<T(* volatile)[N]> {
    using type = typename extract_raw_type<T>::type(*)[N];
};
template <typename T, std::size_t N>
struct extract_raw_type<T(* const volatile)[N]> {
    using type = typename extract_raw_type<T>::type(*)[N];
};
template <typename T, std::size_t N>
struct extract_raw_type<T(&)[N]> {
    using type = typename extract_raw_type<T(*)[N]>::type;
};
template <typename T, std::size_t N>
struct extract_raw_type<T(&&)[N]> {
    using type = typename extract_raw_type<T[N]>::type;
};


// member pointers and member function pointers are disallowed
template <typename T, typename U>
struct extract_raw_type<T U::*>;
template <typename T, typename U>
struct extract_raw_type<T U::* const>;
template <typename T, typename U>
struct extract_raw_type<T U::* volatile>;
template <typename T, typename U>
struct extract_raw_type<T U::* const volatile>;
// member function pointers
template <typename R, typename T, typename... Args>
struct extract_raw_type<R(T::*)(Args...)>;
template <typename R, typename T, typename... Args>
struct extract_raw_type<R(T::* const)(Args...)>;
template <typename R, typename T, typename... Args>
struct extract_raw_type<R(T::* volatile)(Args...)>;
template <typename R, typename T, typename... Args>
struct extract_raw_type<R(T::* const volatile)(Args...)>;

// array types are not decayed to pointers
template <typename T, std::size_t N>
struct extract_raw_type<T[N]> {
    using type = typename extract_raw_type<T>::type[N];
};

// function types are not decayed to pointers
template <typename R, typename... Args>
struct extract_raw_type<R(Args...)> {
    using type = typename extract_raw_type<R>::type (typename extract_raw_type<Args>::type...);
};

/**
 * @brief Removes all cv qualifiers from a pointer type
 * @tparam T The type to remove cv qualifiers from
 * @return The type with all cv qualifiers removed, e.g. `int const volatile*
 * const*` becomes `int**`
 */
template <typename T>
using extract_raw_type_t = typename extract_raw_type<T>::type;

template <typename T, typename = void>
struct is_templated : std::false_type {};
template <template <typename...> typename T, typename... Args>
struct is_templated<T<Args...>, std::void_t<T<Args...>>> : std::true_type {};
template <typename T>
constexpr auto is_templated_v = is_templated<T>::value;

template <typename T>
struct get_template_args {
    using type = std::tuple<>;
};
template <template <typename...> typename T, typename... Args>
struct get_template_args<T<Args...>> {
    using type = std::tuple<Args...>;
};

template <typename T>
using get_template_args_t = typename get_template_args<T>::type;

template <typename T>
struct get_function_params {
    using type = std::tuple<>;
};

template <typename R, typename... Args>
struct get_function_params<R(Args...)> {
    using type = std::tuple<R, Args...>;
};

template <typename T>
using get_function_params_t = typename get_function_params<T>::type;

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

struct for_each {
    template <typename TupleLike, typename Callable>
    static constexpr auto element_in_tuple(TupleLike&& tuple,
                                           Callable&& callable) {
        if constexpr (is_tuple_like_v<std::decay_t<TupleLike>>) {
            std::apply(
                [&callable](auto&&... args) {
                    (callable(std::forward<decltype(args)>(args)), ...);
                },
                std::forward<TupleLike>(tuple));
        } else {
            static_assert(is_tuple_like_v<std::decay_t<TupleLike>>,
                          "for_each requires a tuple-like type");
        }
    }

    template <typename TupleLike, typename Callable>
    static constexpr auto type_in_tuple(Callable&& callable) {
        if constexpr (is_tuple_like_v<std::decay_t<TupleLike>>) {
            type_in_tuple_impl<TupleLike>(std::forward<Callable>(callable),
                                          std::make_index_sequence <
                                              std::tuple_size_v < std::decay_t <
                                              TupleLike >>> {});
        } else {
            static_assert(is_tuple_like_v<std::decay_t<TupleLike>>,
                          "for_each requires a tuple-like type");
        }
    }
   private:
    template <typename TupleLike, typename Callable, std::size_t... Is>
    static constexpr auto type_in_tuple_impl(Callable&& callable,
                                             std::index_sequence<Is...>) {
        // use pointer to avoid instantiation of any types
        (callable(std::add_pointer_t<std::tuple_element_t<Is, std::decay_t<TupleLike>>>{nullptr}), ...);
    }
};

}  // namespace Traits
}  // namespace PTS