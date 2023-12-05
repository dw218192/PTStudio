#pragma once
#include <string_view>

#include "reflection.h"
#include "utils.h"
#include "objectID.h"

namespace PTS {
    struct Scene;

    enum class ObjectConstructorUsage {
        SERIALIZE,
        DEFAULT,
    };
    struct Object {
        // set default arguments because serialization requires a "default" constructor
        Object(ObjectConstructorUsage usage = ObjectConstructorUsage::SERIALIZE) noexcept;
        explicit Object(std::string_view name) noexcept;
        virtual ~Object() noexcept;
        
        NODISCARD auto get_name() const noexcept -> std::string_view {
            return m_name;
        }
        auto set_name(std::string_view name) noexcept {
            m_name = name;
        }
        NODISCARD auto get_id() const noexcept -> ObjectID {
            return m_id;
        }

        auto on_deserialize() noexcept -> void;
        
        BEGIN_REFLECT(Object, void);
        FIELD(std::string, m_name, "Object",
            MSerialize{}, MNoInspect{}); // handled explicitly
        FIELD(ObjectID, m_id, k_invalid_obj_id,
            MSerialize{}, MNoInspect{}); // not editable
        END_REFLECT();

        // enables dynamic retrieval of class info for polymorphic types
        DECL_DYNAMIC_INFO();
    };
} // namespace PTS