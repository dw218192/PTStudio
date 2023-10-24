#pragma once
#include "utils.h"

/**
 * \brief A thin wrapper on top of std::array that allows for strongly typed enum indexing
 * \tparam Enum the enum type to use for indexing
 * \tparam T the type of the array elements
 * \details Assumes the enum class has a _COUNT value as the last value\n
 * and that the enum values are contiguous\n
 * and that the enum values start at 0
*/
template<typename Enum, typename T>
struct EArray {
    using container_type = std::array<T, static_cast<std::size_t>(Enum::__COUNT)>;

    static_assert(std::is_enum_v<Enum>, "Enum must be an enum type");
    static_assert(std::is_default_constructible_v<T>, "T must be default constructible");
    DEFAULT_COPY_MOVE(EArray);

    EArray() = default;
    EArray(std::initializer_list<std::pair<Enum, T>> init) noexcept {
        for (auto const& [e, t] : init) {
            m_data[static_cast<std::size_t>(e)] = std::move(t);
        }
    }

    constexpr auto operator[](Enum e) noexcept -> T& { return m_data[static_cast<std::size_t>(e)]; }
    constexpr auto operator[](Enum e) const noexcept -> T const& { return m_data[static_cast<std::size_t>(e)]; }

    constexpr auto begin() const noexcept { return m_data.begin(); }
    constexpr auto end() const noexcept { return m_data.end(); }
    constexpr auto begin() noexcept { return m_data.begin(); }
    constexpr auto end() noexcept { return m_data.end(); }

    constexpr auto data() noexcept -> T* { return m_data.data(); }
    constexpr auto data() const noexcept -> T const* { return m_data.data(); }

    constexpr auto size() const noexcept -> std::size_t { return m_data.size(); }
    constexpr auto swap(EArray& other) noexcept -> void { m_data.swap(other.m_data); }
private:
    container_type m_data{};
};