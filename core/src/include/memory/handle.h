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
    Handle(Arena& arena, Address addr) : m_addr{addr}, m_arena {&arena} {}
    Handle(Handle const& other) = default;
    Handle(Handle&& other) = default;

    template <typename DerivedOrSelf, typename = std::enable_if_t<std::is_base_of_v<T, DerivedOrSelf>>>
    Handle(Handle<DerivedOrSelf> const& other) : m_addr{other.get_addr()}, m_arena {&other.get_arena()} {}

    auto operator=(Handle const& other) -> Handle& = default;
    auto operator=(Handle&& other) -> Handle& = default;

    auto get_addr() const noexcept -> Address {
        return m_addr;
    }
    auto get_arena() const -> Arena& {
        return *m_arena;
    }
    auto operator->() const -> T* {
        return dynamic_cast<T*>(m_arena->get(
			static_cast<Object*>(m_addr.get())->get_id()
        ));
    }
    auto operator*() const noexcept -> T& {
        return *operator->();
    }
    auto operator==(Handle const& other) const noexcept -> bool {
        return m_addr == other.m_addr && m_arena == other.m_arena;
    }
    auto operator!=(Handle const& other) const noexcept -> bool {
        return !operator==(other);
    }
    auto operator==(T* ptr) const noexcept -> bool {
        return operator->() == ptr;
    }
    auto operator!=(T* ptr) const noexcept -> bool {
        return !operator==(ptr);
    }
    auto operator!() const noexcept -> bool {
        return !m_addr;
    }
    // no implicit conversion to bool because it's error-prone
    auto valid() const noexcept -> bool {
        return !operator!();
    }
    auto get() const -> T* {
        return operator->();
    }

    auto is_alive() const -> bool {
        return m_arena->is_alive(
            static_cast<Object*>(m_addr.get())->get_id()
        );
    }

    template <typename Derived, typename = std::enable_if_t<std::is_base_of_v<T, Derived>>>
    auto as() const -> Handle<Derived> {
        if (dynamic_cast<Derived*>(get())) {
            return Handle<Derived>{*m_arena, m_addr};
        } else {
            return Handle<Derived>{};
        }
    }
    
private:
    Address m_addr {};
    Arena* m_arena {nullptr};
};

} // namespace PTS