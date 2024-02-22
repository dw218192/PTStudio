#pragma once

#include "../utils.h"
#include "memTypes.h"
#include "object/objectID.h"

namespace PTS {

struct Object;

template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
struct Handle;

/**
 * @brief Arena is a memory pool used to allocate and deallocate anything derived from Object.
 * It assigns a unique ID to each object and keeps track of the alive objects.
*/
struct Arena {
    static constexpr auto k_default_block_num = 5;

    static auto get_or_create(size_t id) -> Arena&;
    auto get_id() const noexcept -> ArenaID;
    auto is_alive(ObjectID id) const noexcept -> bool;
    auto is_alive(Handle<Object> const& handle) const noexcept -> bool;
    auto get(ObjectID id) const noexcept -> Object const*;
    auto get(ObjectID id) noexcept -> Object*;
    template <typename T, typename... Args>
    auto allocate(Args&&... args) -> Handle<T>;
    auto deallocate(ObjectID id) noexcept -> void;

private:
    static std::vector<Arena> s_arenas;

    Arena() = default;
    DEFAULT_COPY_MOVE(Arena);

    ArenaID m_id {0};
    // each type of object has its own memory pools
    // when a memory pool is full, another one is created
    std::unordered_map<std::type_index, FixedSizePoolAllocator> m_memory;
    std::unordered_map<ObjectID, Address> m_alive_objects;
};

} // namespace PTS