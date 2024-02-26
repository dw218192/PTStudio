#pragma once

#include <typeindex>

#include "_arena_fwd.h"

#include "fixedSizePoolAllocator.h"
#include "handle.h"
#include "object/object.h"

template <typename T, typename... Args>
auto PTS::Arena::allocate(Args&&... args) -> Handle<T>  {
    static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");

    auto const idx = std::type_index{ typeid(T) };
    auto it = m_memory.find(idx);
    if (it == m_memory.end()) {
        auto const res = m_memory.emplace(idx, FixedSizePoolAllocator{ sizeof(T) });
        it = res.first;
    }
    auto& allocator = it->second;

    // allocate memory and initialize object
    auto const addr = allocator.allocate();
    auto obj = new (addr.get()) T(std::forward<Args>(args)...);
    auto const as_base = static_cast<Object*>(obj);
    as_base->m_arena_id = m_id;
    as_base->m_id = ObjectIDGenerator::generate_id();

    ObjectIDGenerator::register_id(as_base->m_id);
    auto id = obj->get_id();
    m_alive_objects[id] = addr;

    return Handle<T>{ *this, addr };
}