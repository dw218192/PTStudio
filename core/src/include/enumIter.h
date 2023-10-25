#pragma once
#include "utils.h"

/**
 * \brief Iterator that iterates over all values of an enum class
 * \tparam Enum the enum class to iterate over
 * \details NOT meant to be general purpose\n
 * Assumes the enum class has a _COUNT value as the last value\n
 * and that the enum values are contiguous\n
 * and that the enum values start at 0
*/
template<typename Enum>
struct EIter {
    struct Iterator {
        DEFAULT_COPY_MOVE(Iterator);
        Iterator(Enum val) noexcept : m_val{ val } {}

        constexpr auto operator*() const noexcept -> Enum { return m_val; }
        constexpr auto operator++() noexcept -> Iterator& {
            m_val = static_cast<Enum>(static_cast<int>(m_val) + 1); 
            return *this; 
        }
        constexpr auto operator!=(Iterator const& other) const noexcept -> bool { return m_val != other.m_val; }
        
    private:
        Enum m_val;
    };
    constexpr static auto begin() noexcept -> Iterator { return Iterator{ static_cast<Enum>(0) }; }
    constexpr static auto end() noexcept -> Iterator { return Iterator{ Enum::__COUNT }; }
};
