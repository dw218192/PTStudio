#pragma once

#include <imgui.h>
#include <ImGuizmo.h>

#include "transform.h"
#include "material.h"

namespace ImGui {
    bool TransformField(const char* label, Transform& transform, ImGuizmo::OPERATION& op, ImGuizmo::MODE& mode, bool& snap, glm::vec3& snap_scale);
    bool MaterialField(const char* label, Material& material);
}