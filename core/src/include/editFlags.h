#pragma once

namespace PTS {
/**
 * \brief Represents if an editable is visible in the scene or can be edited in the editor
*/
enum EditFlags {
    None = 0,
    Visible = 1 << 0,
    Editable = 1 << 1,
};

auto constexpr edit_flags_modifier = MEnumFlags{
    2,
    [](int idx) -> char const* {
        switch (idx) {
        case 0: return "Visible";
        case 1: return "Editable";
        default: return "Unknown";
        }
    }
};
} // namespace PTS