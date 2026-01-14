#pragma once

#include <array>
#include <iostream>
#include <string_view>

/**
 * @brief Compile time string manipulation utilities.
 */
namespace PTS {

// from
// https://stackoverflow.com/questions/38955940/how-to-concatenate-static-strings-at-compile-time/62823211#62823211
template <std::string_view const&... Strs>
struct join {
    // Join all strings into a single std::array of chars
    static constexpr auto impl() noexcept {
        constexpr std::size_t len = (Strs.size() + ... + 0);
        std::array<char, len + 1> arr{};
        auto append = [i = 0, &arr](auto const& s) mutable {
            for (auto c : s) arr[i++] = c;
        };
        (append(Strs), ...);
        arr[len] = 0;
        return arr;
    }

    // Give the joined string static storage
    static constexpr auto arr = impl();
    // View as a std::string_view
    static constexpr std::string_view value{arr.data(), arr.size() - 1};
};

// Helper to get the value out
template <std::string_view const&... Strs>
static constexpr auto join_v = join<Strs...>::value;

// to_string functions
template <int val>
struct to_str {
    using buffer = std::array<char, 33>;

    static constexpr auto impl() noexcept -> std::tuple<buffer, int, int> {
        auto val_copy = val;
        buffer arr{};
        auto start = 1;
        if (val < 0) {
            arr[0] = '-';
            val_copy = -val_copy;
            start = 0;
        }

        auto it = arr.begin() + 1;
        do {
            *it++ = '0' + val_copy % 10;
            val_copy /= 10;
        } while (val_copy != 0);

        // reverse the range [arr.begin() + 1, it)
        for (auto i = arr.begin() + 1, j = it - 1; i < j; ++i, --j) {
            auto tmp = *i;
            *i = *j;
            *j = tmp;
        }
        return {arr, start, static_cast<int>(it - arr.begin() - start)};
    }

    static constexpr auto arr = impl();
    static constexpr std::string_view value{std::get<0>(arr).data() + std::get<1>(arr),
                                            std::get<2>(arr)};
};

template <int val>
static constexpr auto to_str_v = to_str<val>::value;
}  // namespace PTS
