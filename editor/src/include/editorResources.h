#pragma once

#include "editor/generated/embedded_resources.h"

namespace pts {
constexpr auto k_editor_tutorial_text = R"(This is a simple editor.
Basic Operations:
- Left click to select object
- Left click + drag to move/rotate/scale with gizmo
- Right click + drag to orbit camera
- Middle click + drag to pan camera
- Scroll to zoom
- Press Escape to deselect object
- Press Delete to delete selected object
- Press F to focus on selected object
- Press W/E/R to switch between translate/rotate/scale gizmo
- Press X to toggle snap
)";
}  // namespace pts
