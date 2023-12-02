#pragma once
#include <string_view>

#include "reflection.h"
#include "utils.h"

namespace PTS {
    struct Scene;

    struct Object {
        Object() noexcept = default;
        Object(std::string_view name) noexcept : m_name(name) {}
        virtual ~Object() = default;
        
        NODISCARD auto get_name() const noexcept -> std::string_view {
            return m_name;
        }
        auto set_name(std::string_view name) noexcept {
            m_name = name;
        }
        
        BEGIN_REFLECT(Object);
        FIELD_MOD(std::string, m_name, "Object",
            MSerialize{}, MNoInspect{}); // handled explicitly
        END_REFLECT();
    };
} // namespace PTS