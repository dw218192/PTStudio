#include "memory/fixedSizePoolAllocator.h"
#include "memory/arena.h"
#include <utility> //std::max

#ifndef NDEBUG
#include <cassert>
#define ASSERT(...) assert(__VA_ARGS__)
#else
#define ASSERT(...) ((void)0)
#endif

auto PTS::Address::get() const noexcept -> void* {
    if (!m_allocator) {
        return nullptr;
    }
    return m_allocator->addr_from_index(m_offset);
}

auto PTS::Address::valid() const noexcept -> bool {
    return m_allocator != nullptr && m_allocator->m_is_used[m_offset];
}

auto PTS::Address::deallocate() noexcept -> void {
    if (valid()) {
        m_allocator->deallocate(*this);
        m_allocator = nullptr;
        m_offset = 0;
    }

    // post-condition
    ASSERT(!valid());
}

PTS::FixedSizePoolAllocator::FixedSizePoolAllocator(size_t block_size) noexcept
    : m_block_size { std::max(block_size, sizeof(size_t)) },
    m_num_used {0},
    m_num_initialized {0},
    m_free_list {0},
	m_data(m_block_size * Arena::k_default_block_num),
	m_is_used(Arena::k_default_block_num) {}

PTS::FixedSizePoolAllocator::~FixedSizePoolAllocator() noexcept = default;

auto PTS::FixedSizePoolAllocator::get_tot_block_num() const noexcept -> size_t {
    ASSERT(m_is_used.size() == m_data.size() / m_block_size);
    return m_is_used.size();
}

auto PTS::FixedSizePoolAllocator::addr_from_index(size_t index) -> void* {
    ASSERT(index < m_is_used.size());
    return m_data.data() + index * m_block_size;
}

auto PTS::FixedSizePoolAllocator::index_from_addr(void* addr) const -> size_t {
    return static_cast<size_t>(
        (static_cast<Byte*>(addr) - m_data.data()) / m_block_size
    );
}

auto PTS::FixedSizePoolAllocator::allocate() -> Address {
    if (m_num_used == get_tot_block_num()) {
        // expand capacity if full
        m_data.resize(m_data.size() * 2);
        m_is_used.resize(m_data.size() / m_block_size);
    }

    if (m_num_initialized < get_tot_block_num()) {
        auto const next = static_cast<size_t*>(addr_from_index(m_num_initialized));
        *next = ++ m_num_initialized;
    }

    // precondition
    ASSERT(m_num_used < m_num_initialized);
    ASSERT(m_free_list < m_num_initialized);

    ++ m_num_used;

    // pop the first free block
    auto ret = Address {};
    ret.m_allocator = this;
    ret.m_offset = m_free_list;
    m_is_used[ret.m_offset] = true;

    m_free_list = *static_cast<size_t*>(addr_from_index(m_free_list));

    // postcondition
    ASSERT(m_num_used <= m_num_initialized);
    ASSERT(m_free_list <= m_num_initialized);

    return ret;
}

auto PTS::FixedSizePoolAllocator::deallocate(Address ptr) noexcept -> void {
    if (ptr.m_allocator != this) {
        return;
    }
    if (ptr.m_offset >= m_num_initialized) {
        return;
    }
    if (!m_is_used[ptr.m_offset]) {
        return;
    }

    // precondition
    ASSERT(m_num_used > 0);
    ASSERT(m_free_list <= m_num_initialized);

	-- m_num_used;
    m_is_used[ptr.m_offset] = false;

    // prepend the block to the free list
    *static_cast<size_t*>(addr_from_index(ptr.m_offset)) = m_free_list;
    m_free_list = ptr.m_offset;

    // postcondition
    ASSERT(m_num_used < m_num_initialized);
    ASSERT(m_free_list < m_num_initialized);
}
