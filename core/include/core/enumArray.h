#pragma once
#include <array>
#include <type_traits>

#include "utils.h"

namespace PTS {
/**
 * \brief A thin wrapper on top of std::array that allows for strongly typed enum indexing
 * \tparam Enum the enum type to use for indexing
 * \tparam T the type of the array elements
 * \details NOT meant to be general purpose\n
 * Assumes the enum class has a __COUNT value as the last value\n
 * and that the enum values are contiguous\n
 * and that the enum values start at 0
 */
template <typename Enum, typename T,
          typename = std::enable_if_t<std::is_enum_v<Enum>, std::void_t<decltype(Enum::__COUNT)>>>
struct EArray {
    struct PairViewIterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<Enum, T>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        constexpr PairViewIterator() noexcept = default;
        constexpr PairViewIterator(T* data, std::size_t index) noexcept
            : m_data(data), m_index(index) {
        }

        constexpr auto operator++() noexcept -> PairViewIterator& {
            ++m_index;
            return *this;
        }

        constexpr auto operator++(int) noexcept -> PairViewIterator {
            auto copy = *this;
            ++m_index;
            return copy;
        }

        constexpr auto operator*() noexcept -> value_type {
            return {static_cast<Enum>(m_index), m_data[m_index]};
        }

        constexpr auto operator*() const noexcept -> value_type const {
            return {static_cast<Enum>(m_index), m_data[m_index]};
        }

        constexpr auto operator==(PairViewIterator const& other) const noexcept -> bool {
            return m_data == other.m_data && m_index == other.m_index;
        }

        constexpr auto operator!=(PairViewIterator const& other) const noexcept -> bool {
            return !(*this == other);
        }

       private:
        T* m_data{nullptr};
        std::size_t m_index{0};
    };

    struct PairView {
        constexpr PairView(EArray const& arr) noexcept : m_data(arr.m_data) {
        }
        constexpr auto begin() const noexcept -> PairViewIterator {
            return {m_data.data(), 0};
        }
        constexpr auto end() const noexcept -> PairViewIterator {
            return {m_data.data(), m_data.size()};
        }

       private:
        std::array<T, static_cast<std::size_t>(Enum::__COUNT)> const& m_data;
    };

    using container_type = std::array<T, static_cast<std::size_t>(Enum::__COUNT)>;

    static_assert(std::is_enum_v<Enum>, "Enum must be an enum type");
    static_assert(std::is_default_constructible_v<T>, "T must be default constructible");
    DEFAULT_COPY_MOVE(EArray);

    constexpr EArray() = default;

    constexpr EArray(std::initializer_list<std::pair<Enum, T>> init) noexcept {
        for (auto const& [e, t] : init) {
            m_data[static_cast<std::size_t>(e)] = std::move(t);
        }
    }

    constexpr auto operator[](Enum e) noexcept -> T& {
        return m_data[static_cast<std::size_t>(e)];
    }
    constexpr auto operator[](Enum e) const noexcept -> T const& {
        return m_data[static_cast<std::size_t>(e)];
    }

    constexpr auto begin() const noexcept {
        return m_data.begin();
    }
    constexpr auto end() const noexcept {
        return m_data.end();
    }
    constexpr auto begin() noexcept {
        return m_data.begin();
    }
    constexpr auto end() noexcept {
        return m_data.end();
    }

    constexpr auto data() noexcept -> T* {
        return m_data.data();
    }
    constexpr auto data() const noexcept -> T const* {
        return m_data.data();
    }

    constexpr auto size() const noexcept -> std::size_t {
        return m_data.size();
    }
    constexpr auto swap(EArray& other) noexcept -> void {
        m_data.swap(other.m_data);
    }

    constexpr auto pair_view() const noexcept -> PairView {
        return {*this};
    }

   private:
    container_type m_data{};
};
}  // namespace PTS
