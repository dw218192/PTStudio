#pragma once
#include <boost/algorithm/string/predicate.hpp>
#include <boost/describe/enum.hpp>
#include <boost/describe/enum_to_string.hpp>
#include <boost/describe/enumerators.hpp>
#include <boost/mp11/algorithm.hpp>
#include <optional>
#include <string_view>

namespace pts {
/**
 * @brief Attempts to parse a string to a boost-described enum.
 * @param s The string to convert.
 * @param case_sensitive Whether to perform case-sensitive comparison.
 * @return The enum value, or std::nullopt if the string is not a valid enum value.
 * @note This function is case-insensitive by default.
 */
template <class E>
inline std::optional<E> from_string(std::string_view s, bool case_sensitive = false) {
    std::optional<E> out;
    boost::mp11::mp_for_each<boost::describe::describe_enumerators<E>>([&](auto d) {
        if (!out && (case_sensitive ? s == d.name : boost::iequals(s, d.name)))
            out = static_cast<E>(d.value);
    });
    return out;
}

/**
 * @brief Converts a boost-described enum to its string name.
 * @param e The enum value to convert.
 * @param default_name The string to return if the enum value is not found.
 * @return The string name of the enum value, or default_name if not found.
 */
template <class E>
inline const char* to_string(E e, const char* default_name = "UNKNOWN") {
    return boost::describe::enum_to_string(e, default_name);
}
}  // namespace pts