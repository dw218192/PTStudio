#include "memory/arena.h"

auto PTS::Arena::get_or_create(size_t id) -> Arena& {
    if (id < detail::s_arenas.size()) {
        return detail::s_arenas[id].val;
    } else {
        // create a new arena
        detail::s_arenas.resize(id + 1);
        return detail::s_arenas[id].val;
    }
}

auto PTS::Arena::get_id() const noexcept -> ArenaID {
    return m_id;
}

auto PTS::Arena::is_alive(ObjectID id) const noexcept -> bool {
    return m_alive_objects.count(id);
}

auto PTS::Arena::is_alive(Handle<Object> const& handle) const noexcept -> bool {
    return is_alive(handle->get_id());
}

auto PTS::Arena::get(ObjectID id) const noexcept -> Object const* {
    auto it = m_alive_objects.find(id);
    if (it != m_alive_objects.end()) {
        return static_cast<Object const*>(it->second.get());
    } else {
        return nullptr;
    }
}

auto PTS::Arena::get(ObjectID id) noexcept -> Object* {
    return const_cast<Object*>(static_cast<Arena const*>(this)->get(id));
}

auto PTS::Arena::deallocate(ObjectID id) noexcept -> void {
    auto const it = m_alive_objects.find(id);
    if (it != m_alive_objects.end()) {
        auto* obj = static_cast<Object*>(it->second.get());
        obj->~Object();

        it->second.deallocate();
        m_alive_objects.erase(it);
    }
}

auto PTS::Arena::deallocate(Handle<Object> const& handle) noexcept -> void {
    if (handle.is_alive()) {
        deallocate(handle->get_id());
    }
}
