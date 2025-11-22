#pragma once
#include "UniformVar.h"
#include "includes.h"
#include "material.h"
#include "reflection.h"
#include "transform.h"

namespace ImGui {
bool TransformField(const char* label, PTS::Transform& transform, ImGuizmo::OPERATION& op,
                    ImGuizmo::MODE& mode, bool& snap, glm::vec3& snap_scale);
bool ShaderVariableField(const char* label, PTS::UniformVar& variable);
}  // namespace ImGui