#pragma once
#include <imgui.h>

#include <glm/glm.hpp>

// math for ImVec2
inline ImVec2 operator+(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return ImVec2{lhs.x + rhs.x, lhs.y + rhs.y};
}
inline ImVec2 operator-(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return ImVec2{lhs.x - rhs.x, lhs.y - rhs.y};
}
inline ImVec2 operator*(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return ImVec2{lhs.x * rhs.x, lhs.y * rhs.y};
}
inline ImVec2 operator*(ImVec2 const& lhs, float rhs) noexcept {
    return ImVec2{lhs.x * rhs, lhs.y * rhs};
}
inline ImVec2 operator*(float lhs, ImVec2 const& rhs) noexcept {
    return ImVec2{lhs * rhs.x, lhs * rhs.y};
}
inline ImVec2 operator/(ImVec2 const& lhs, ImVec2 const& rhs) noexcept {
    return ImVec2{lhs.x / rhs.x, lhs.y / rhs.y};
}
inline ImVec2 operator/(ImVec2 const& lhs, float rhs) noexcept {
    return ImVec2{lhs.x / rhs, lhs.y / rhs};
}
inline ImVec2 operator/(float lhs, ImVec2 const& rhs) noexcept {
    return ImVec2{lhs / rhs.x, lhs / rhs.y};
}
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