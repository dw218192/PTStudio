#pragma once
#include <string_view>

#include "reflection.h"
#include "utils.h"
#include "objectID.h"
#include "memory/memTypes.h"

namespace PTS {
    struct Scene;
    struct Archive;
    struct Arena;

    /**
     * @brief Base class for all objects in the engine;
     * Provides basic functionality for 
     * - Serialization
     * - Deserialization
     * - UUID
     * - Name
     * - Static and dynamic reflection
    */
    struct Object {
        friend struct Arena;

        DEFAULT_COPY_MOVE(Object);

        // set default arguments because serialization requires a "default" constructor
        explicit Object(std::string_view name) noexcept;
        
        NODISCARD auto get_arena() -> Arena& {
            return const_cast<Arena&>(static_cast<Object const*>(this)->get_arena());
        }
        NODISCARD auto get_arena() const->Arena const&;
        NODISCARD auto get_name() const noexcept -> std::string_view {
            return m_name;
        }
        auto set_name(std::string_view name) noexcept {
            m_name = name;
        }
        NODISCARD auto get_id() const noexcept -> ObjectID {
            return m_id;
        }

        virtual auto serialize(Archive& archive) const -> void;
        virtual auto on_deserialize() noexcept -> void;

    protected:
        BEGIN_REFLECT(Object, void);
        FIELD(std::string, m_name, "Object",
            MSerialize{}, MNoInspect{}); // handled explicitly
        FIELD(ObjectID, m_id, k_invalid_obj_id,
            MSerialize{}, MNoInspect{}); // not editable
        END_REFLECT();

        // enables dynamic retrieval of class info for polymorphic types
        DECL_DYNAMIC_INFO();

    protected:
        virtual ~Object() noexcept;

    private:
        ArenaID m_arena_id;
    };
} // namespace PTS