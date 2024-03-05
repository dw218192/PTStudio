#pragma once
#include <cstddef>

namespace PTS {
struct FixedSizePoolAllocator;

struct Address {
    friend struct FixedSizePoolAllocator;
    
    auto get() const noexcept -> void*;
    auto valid() const noexcept -> bool;
    auto deallocate() noexcept -> void;

    auto operator!() const noexcept -> bool {
        return !valid();
    }
    operator bool() const {
        return valid();
    }

    template<typename T>
    auto operator==(T* ptr) const noexcept -> bool {
        return get() == ptr;
    }
    auto operator==(std::nullptr_t) const noexcept -> bool {
        return !valid();
    }
    auto operator!=(std::nullptr_t null) const noexcept -> bool {
        return !operator==(null);
    }
    auto operator==(Address const& other) const noexcept -> bool {
        return m_offset == other.m_offset;
    }
    auto operator!=(Address const& other) const noexcept -> bool {
        return !operator==(other);
    }
    auto operator>(Address const& other) const noexcept -> bool {
        return m_offset > other.m_offset;
    }
    auto operator>=(Address const& other) const noexcept -> bool {
        return operator==(other) || operator>(other);
    }
    auto operator<(Address const& other) const noexcept -> bool {
        return !operator>=(other);
    }
    auto operator<=(Address const& other) const noexcept -> bool {
        return !operator>(other);
    }

private:
    FixedSizePoolAllocator* m_allocator {nullptr}; // always non-null, and not invalidated
    size_t m_offset {0}; // index used by the allocator, effectively a logical address
};

using ArenaID = size_t;
constexpr auto k_invalid_arena_id = ArenaID{ 0xdeadbeef };
}
