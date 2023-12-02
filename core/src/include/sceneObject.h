#pragma once

#include <tl/expected.hpp>
#include <tcb/span.hpp>
#include <string>
#include <optional>

#include "transform.h"
#include "object.h"
#include "reflection.h"
#include "editFlags.h"

namespace PTS {
    struct Scene;

    struct SceneObject : Object {
        SceneObject() noexcept = default;
        SceneObject(Scene const& scene, std::string_view name, Transform transform);
        SceneObject(Scene const& scene, Transform transform);

        NODISCARD auto get_transform() const noexcept -> auto const& {
            return m_transform;
        }
        auto get_edit_flags() const noexcept {
            return m_flags;
        }
        auto set_transform(Transform transform) noexcept {
            m_transform = std::move(transform);
        }
        auto set_edit_flags(int flags) noexcept {
            m_flags = static_cast<EditFlags>(flags);
        }
    private:
        BEGIN_REFLECT_INHERIT(SceneObject, Object);
        FIELD_MOD(Transform, m_transform, {},
            MSerialize{}, MNoInspect{}); // handled explicitly
        FIELD_MOD(EditFlags, m_flags, static_cast<EditFlags>(EditFlags::Visible | EditFlags::Selectable),
            MSerialize{}, edit_flags_modifier);
        FIELD_MOD(ViewPtr<Scene>, m_scene, nullptr,
            MSerialize{}, MNoInspect{}); // not editable
        END_REFLECT_INHERIT();
    };
}