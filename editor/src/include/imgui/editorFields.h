#pragma once
#include <core/legacy/material.h>
#include <core/legacy/reflection.h>
#include <core/legacy/transform.h>
#include <gl_utils/UniformVar.h>

#include "includes.h"

namespace ImGui {
bool TransformField(const char* label, PTS::Transform& transform, ImGuizmo::OPERATION& op,
                    ImGuizmo::MODE& mode, bool& snap, glm::vec3& snap_scale);
bool ShaderVariableField(const char* label, PTS::UniformVar& variable);
}  // namespace ImGui