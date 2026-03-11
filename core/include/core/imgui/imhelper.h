#pragma once
// imgui math operators are enabled via IMGUI_DEFINE_MATH_OPERATORS from imguizmo package
#include <imgui.h>

#include <glm/glm.hpp>

// Additional comparison operators for ImVec2 (not provided by IMGUI_DEFINE_MATH_OPERATORS)
inline bool operator<(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return lhs.x < rhs.x && lhs.y < rhs.y;
}
inline bool operator>(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return lhs.x > rhs.x && lhs.y > rhs.y;
}
inline bool operator<=(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return lhs.x <= rhs.x && lhs.y <= rhs.y;
}
inline bool operator>=(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return lhs.x >= rhs.x && lhs.y >= rhs.y;
}

// convert ImVec2 to glm::vec2
inline glm::vec2 to_glm(ImVec2 const& v) noexcept {
    return glm::vec2{v.x, v.y};
}