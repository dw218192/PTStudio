#pragma once

#include "utils.h"
#include "object.h"
#include "objectID.h"
#include "objectHandle.h"

namespace PTS {
/**
 * @brief Arena is a memory pool used to allocate and deallocate anything derived from Object.
 * It assigns a unique ID to each object and keeps track of the alive objects.
 * for now, it just wraps new and delete, but it will be used to implement object pooling in the future.
*/
struct Arena {
    auto is_alive(ObjectID id) const noexcept -> bool {
        return m_alive_objects.count(id);
    }
    auto is_alive(Handle<Object> const& handle) const noexcept -> bool {
        return is_alive(handle->get_id());
    }

    template <typename T, typename... Args>
    auto allocate(Args&&... args) -> Handle<T> {
        static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
        
        auto* obj = new T{std::forward<Args>(args)...};
        auto as_base = static_cast<Object*>(obj);
        as_base->m_arena = this;
        as_base->m_id = ObjectIDGenerator::generate_id();
        ObjectIDGenerator::register_id(as_base->m_id);
        
        auto id = obj->get_id();
        m_alive_objects[id] = obj;
        return Handle<T>{obj};
    }

    auto deallocate(ObjectID id) noexcept -> void {
        auto it = m_alive_objects.find(id);
        if (it != m_alive_objects.end()) {
            ObjectIDGenerator::unregister_id(id);

            delete it->second.get();
            m_alive_objects.erase(it);
        }
    }

private:
    std::unordered_map<ObjectID, Handle<Object>> m_alive_objects;
};

} // namespace PTS