#pragma once

namespace PTS {
struct Object;
/**
 * @brief Handle to an object in the scene.
 * The handle is basically a reference that is valid as long as the object is alive.
*/
template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
struct Handle {
    Handle(T* object) : m_object {object} {}

    template <typename Base, typename = std::enable_if_t<std::is_base_of_v<Base, T>>>
    Handle(Handle<Base> const& other) : m_object {other.get()} {}

    auto operator->() const -> T* {
        return m_object;
    }
    auto operator*() const -> T& {
        return *m_object;
    }
    auto operator==(Handle const& other) const -> bool {
        return m_object == other.m_object;
    }
    auto operator!=(Handle const& other) const -> bool {
        return m_object != other.m_object;
    }
    auto operator==(T* ptr) const -> bool {
        return m_object == ptr;
    }
    auto operator!=(T* ptr) const -> bool {
        return m_object != ptr;
    }
    auto operator==(std::nullptr_t) const -> bool {
        return m_object == nullptr;
    }
    auto operator!=(std::nullptr_t) const -> bool {
        return m_object != nullptr;
    }
    auto operator!() const -> bool {
        return m_object == nullptr;
    }
    operator bool() const {
        return m_object != nullptr;
    }
    auto reset(T* object) -> void {
        m_object = object;
    }
    auto get() const -> T* {
        return m_object;
    }
    auto is_alive() const -> bool {
        return m_object != nullptr && m_object->get_arena().is_alive(m_object->get_id());
    }
    template <typename Derived, typename = std::enable_if_t<std::is_base_of_v<T, Derived>>>
    auto as() const -> Handle<Derived> {
        return Handle<Derived>{ dynamic_cast<Derived*>(m_object) };
    }
    
private:
    T* m_object;
};

} // namespace PTS