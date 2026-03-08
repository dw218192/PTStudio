#pragma once

#include "editor/generated/embedded_resources.h"

namespace pts {
constexpr auto k_editor_tutorial_text = R"(This is a simple editor.
Basic Operations:
- Left click to select object
- Press Escape to deselect object
- Press Delete to delete selected object
- Press F to focus on selected object
- Right click to rotate camera
- Left click + drag for dolly and track
- Middle click + drag for pedestal
- Press W/E/R to switch between translate/rotate/scale gizmo
- Press X to toggle snap
)";
}  // namespace pts
