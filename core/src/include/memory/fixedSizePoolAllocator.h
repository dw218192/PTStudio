#pragma once

#include "memTypes.h"
#include <vector>

namespace PTS {

struct FixedSizePoolAllocator {
    friend struct Address;
    using Byte = unsigned char;

    FixedSizePoolAllocator(size_t block_size) noexcept;
    ~FixedSizePoolAllocator() noexcept;
    auto allocate() -> Address;
    auto deallocate(Address ptr) noexcept -> void;
    auto is_full() const noexcept -> bool;

private:
    auto addr_from_index(size_t index) -> void*;
    auto index_from_addr(void* addr) -> size_t;
    size_t m_block_size;
    size_t m_num_used;
    size_t m_num_initialized;
    std::vector<Byte> m_data;
    size_t m_free_list; // index of the first free block in m_data
};

} // namespace PTS