#include "memory/fixedSizePoolAllocator.h"
#include "memory/arena.h"

auto PTS::Address::get() const noexcept -> void* {
    return m_allocator->addr_from_index(m_offset);
}

auto PTS::Address::valid() const noexcept -> bool {
    return m_allocator != nullptr;
}

auto PTS::Address::deallocate() noexcept -> void {
    if (valid()) {
        m_allocator->deallocate(*this);
        m_allocator = nullptr;
        m_offset = 0;
    }
}

PTS::FixedSizePoolAllocator::FixedSizePoolAllocator(size_t block_size) noexcept
    : m_block_size {block_size}, 
    m_num_used {0},
    m_num_initialized {0},
    m_free_list {0}
{
    m_data.resize(block_size * Arena::k_default_block_num);
}

PTS::FixedSizePoolAllocator::~FixedSizePoolAllocator() noexcept = default;

auto PTS::FixedSizePoolAllocator::addr_from_index(size_t index) -> void* {
    auto ret = m_data.data() + index * m_block_size;
    return ret;
}

auto PTS::FixedSizePoolAllocator::index_from_addr(void* addr) -> size_t {
    return static_cast<size_t>(
        (static_cast<Byte*>(addr) - m_data.data()) / m_block_size
    );
}

auto PTS::FixedSizePoolAllocator::allocate() -> Address {
    if (is_full()) {
        return {};
    }

    if (m_num_initialized < m_data.size() / m_block_size) {
        // lazily initialize the memory
        m_data.resize(m_data.size() + m_block_size);
        auto p = static_cast<size_t*>(addr_from_index(m_num_initialized));
        *p = m_num_initialized + 1;
        ++ m_num_initialized;
    }

    ++ m_num_used;

    auto ret = Address {};
    ret.m_allocator = this;
    // pop the first free block
    ret.m_offset = m_free_list;
    m_free_list = *static_cast<size_t*>(addr_from_index(m_free_list));
    return ret;
}

auto PTS::FixedSizePoolAllocator::deallocate(Address ptr) noexcept -> void {
    if (ptr.m_allocator != this) {
        return;
    }

    -- m_num_used;

    // prepend the block to the free list
    *static_cast<size_t*>(addr_from_index(ptr.m_offset)) = m_free_list;
    m_free_list = ptr.m_offset;
}

auto PTS::FixedSizePoolAllocator::is_full() const noexcept -> bool {
    return m_num_used == m_data.size() / m_block_size;
}

