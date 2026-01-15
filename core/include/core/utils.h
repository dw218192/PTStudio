#include <boost/algorithm/string/predicate.hpp>
#include <boost/describe/enum.hpp>
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
}  // namespace pts