#pragma once

namespace PTS {
struct FixedSizePoolAllocator;

struct Address {
    friend struct FixedSizePoolAllocator;

    auto get() const noexcept -> void*;
    auto valid() const noexcept -> bool;
    auto deallocate() noexcept -> void;

private:
    FixedSizePoolAllocator* m_allocator {nullptr}; // always non-null, and not invalidated
    size_t m_offset {0}; // index used by the allocator, effectively a logical address
};

using ArenaID = size_t;
}