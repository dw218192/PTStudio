#pragma once

#include "editor/generated/embedded_resources.h"

namespace pts {
constexpr auto k_editor_tutorial_text = R"(This is a simple editor.
Camera:
- Right click + drag to orbit
- Middle click + drag to pan
- Scroll to zoom
- Hold right click + WASD to fly, Q/E for down/up

Object Manipulation:
- Left click to select, Escape to deselect
- Left click + drag gizmo to transform
- W/E/R to switch translate/rotate/scale gizmo
- X to toggle snap
- Delete to delete selected object
- F to focus on selected object
)";
}  // namespace pts
