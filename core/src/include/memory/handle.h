#pragma once
#include "object/object.h"
#include "object/objectID.h"
#include "_arena_fwd.h"

namespace PTS {
/**
 * @brief Handle to a managed object.
 * The handle is basically a reference that is valid as long as the object is alive.
*/
template <typename T, typename>
struct Handle {
    Handle() = default;
    Handle(Arena& arena, ObjectID id) : m_id {id}, m_arena {&arena} {}
    Handle(std::nullptr_t) : m_id {k_invalid_obj_id}, m_arena {nullptr} {}

    template <typename Base, typename = std::enable_if_t<std::is_base_of_v<Base, T>>>
    Handle(Handle<Base> const& other) : m_id {other.get_id()}, m_arena {other.get_arena()} {}

    auto operator=(Handle const& other) -> Handle& = default;
    auto operator=(Handle&& other) -> Handle& = default;
    auto operator=(std::nullptr_t) -> Handle& {
        m_id = k_invalid_obj_id;
        m_arena = nullptr;
        return *this;
    }
    auto operator=(T* ptr) -> Handle& {
        m_id = ptr->get_id();
        m_arena = ptr->get_arena();
        return *this;
    }

    auto get_id() const -> ObjectID {
        return m_id;
    }
    auto get_arena() const -> Arena& {
        return *m_arena;
    }
    auto operator->() const -> T* {
        return m_arena->get(m_id);
    }
    auto operator*() const -> T& {
        return *operator->();
    }
    auto operator==(Handle const& other) const -> bool {
        return m_id == other.m_id && m_arena == other.m_arena;
    }
    auto operator!=(Handle const& other) const -> bool {
        return !(*this == other);
    }
    auto operator==(T* ptr) const -> bool {
        return operator->() == ptr;
    }
    auto operator!=(T* ptr) const -> bool {
        return !(*this == ptr);
    }
    auto operator==(std::nullptr_t) const -> bool {
        return m_id == k_invalid_obj_id;
    }
    auto operator!=(std::nullptr_t) const -> bool {
        return !(*this == nullptr);
    }
    auto operator!() const -> bool {
        return operator==(nullptr);
    }
    operator bool() const {
        return !operator!();
    }
    auto get() const -> T* {
        return m_arena->get(m_id);
    }
    auto is_alive() const -> bool {
        return m_arena->is_alive(m_id);
    }

    template <typename Derived, typename = std::enable_if_t<std::is_base_of_v<T, Derived>>>
    auto as() const -> Handle<Derived> {
        if (dynamic_cast<Derived*>(get())) {
            return Handle<Derived>{m_arena, m_id};
        } else {
            return Handle<Derived>{};
        }
    }
    
private:
    ObjectID m_id {k_invalid_obj_id};
    Arena* m_arena {nullptr};
};

} // namespace PTS