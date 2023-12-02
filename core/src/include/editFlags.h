#pragma once

namespace PTS {
/**
 * \brief Represents if an editable is visible in the scene or can be edited in the editor
*/
enum EditFlags {
    None = 0,
    Visible = 1 << 0,
    Selectable = 1 << 1, // can be selected in the scene view by ray casting
    // note: even if an object is not selectable, it can still be selected in the object tree

    // internal flags, not exposed to the user
    _NoEdit = 1 << 2, // disallows any editing, used for objects managed by another object
};

auto constexpr edit_flags_modifier = MEnumFlags{
    2,
    [](int idx) -> char const* {
        switch (idx) {
        case 0: return "Visible";
        case 1: return "Selectable";
        default: return "Unknown";
        }
    }
};
} // namespace PTS