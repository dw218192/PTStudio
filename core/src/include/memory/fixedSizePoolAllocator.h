#pragma once

#include "memTypes.h"
#include <vector>

#include "utils.h"

namespace PTS {

struct FixedSizePoolAllocator {
#ifdef UNIT_TEST
    friend struct Accessor;
#endif
	friend struct Address;
	using Byte = unsigned char;

    FixedSizePoolAllocator(size_t block_size) noexcept;
    ~FixedSizePoolAllocator() noexcept;
    DEFAULT_COPY_MOVE(FixedSizePoolAllocator);

    auto allocate() -> Address;
    auto deallocate(Address ptr) noexcept -> void;

#ifdef UNIT_TEST
struct Accessor {
    static auto addr_from_index(FixedSizePoolAllocator& alloc, size_t index) -> void* {
        return alloc.addr_from_index(index);
    }
    static auto index_from_addr(FixedSizePoolAllocator& alloc, void* addr) -> size_t {
        return alloc.index_from_addr(addr);
    }
    static auto num_used(FixedSizePoolAllocator& alloc) -> size_t {
        return alloc.m_num_used;
    }
    static auto num_initialized(FixedSizePoolAllocator& alloc) -> size_t {
        return alloc.m_num_initialized;
    }
    static auto free_list(FixedSizePoolAllocator& alloc) -> size_t {
        return alloc.m_free_list;
    }
    static auto data(FixedSizePoolAllocator& alloc) -> std::vector<Byte>& {
        return alloc.m_data;
    }
    static auto block_size(FixedSizePoolAllocator& alloc) -> size_t {
        return alloc.m_block_size;
    }
};
#endif

private:
    auto get_tot_block_num() const noexcept -> size_t;
    auto addr_from_index(size_t index) -> void*;
    auto index_from_addr(void* addr) const -> size_t;
    size_t m_block_size;
    size_t m_num_used;
    size_t m_num_initialized;
    size_t m_free_list; // index of the first free block in m_data

    std::vector<Byte> m_data;
    std::vector<bool> m_is_used;
};



} // namespace PTS