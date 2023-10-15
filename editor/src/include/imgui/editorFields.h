#pragma once

#include "transform.h"
#include "material.h"

namespace ImGui {
    bool TransformField(const char* label, Transform& transform);
    bool MaterialField(const char* label, Material& material);
}