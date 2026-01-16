#pragma once
#include <core/legacy/transform.h>

#include "includes.h"

namespace ImGui {
    bool TransformField(const char* label, PTS::Transform& transform, ImGuizmo::OPERATION& op,
                        ImGuizmo::MODE& mode, bool& snap, glm::vec3& snap_scale);
}  // namespace ImGui