#pragma once

#include "../utils.h"
#include "memTypes.h"
#include "object/objectID.h"
#include <typeindex>

namespace PTS {
	namespace detail {
		struct ArenaWrapper;
	}

	struct Object;

template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
struct Handle;

/**
 * @brief Arena is a memory pool used to allocate and deallocate anything derived from Object.
 * It assigns a unique ID to each object and keeps track of the alive objects.
*/
struct Arena {
    friend struct detail::ArenaWrapper;
    static constexpr auto k_default_block_num = 5;
    static auto get_arena(size_t id) -> Arena*;
    NODISCARD static auto get_or_create_arena(size_t id) -> Arena&;
    NODISCARD auto get_id() const noexcept -> ArenaID;
    NODISCARD auto is_alive(ObjectID id) const noexcept -> bool;
    NODISCARD auto is_alive(Handle<Object> const& handle) const noexcept -> bool;
    NODISCARD auto get(ObjectID id) const noexcept -> Object const*;
    NODISCARD auto get(ObjectID id) noexcept -> Object*;
    template <typename T, typename... Args>
    NODISCARD auto allocate(Args&&... args) -> Handle<T>;
    auto deallocate(ObjectID id) noexcept -> void;
    auto deallocate(Handle<Object> const& handle) noexcept -> void;
    ~Arena() = default;

private:
    Arena() = default;
    DEFAULT_COPY_MOVE(Arena);

    ArenaID m_id {0};
    // each type of object has its own memory pools
    // when a memory pool is full, another one is created
    std::unordered_map<std::type_index, FixedSizePoolAllocator> m_memory;
    std::unordered_map<ObjectID, Address> m_alive_objects;
};

namespace detail {
    struct ArenaWrapper {
        ArenaWrapper() = default;
        ArenaWrapper(ArenaWrapper const& other) = default;
        ArenaWrapper(Arena const& val) : val{ val } {}
    	Arena val;
    };

    static inline std::vector<ArenaWrapper> s_arenas;
}
} // namespace PTS