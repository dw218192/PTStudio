#pragma once
#include <any>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pts::rendering {

enum class PropertyTag : uint8_t {
    None = 0,
    Color = 1 << 0,
    ReadOnly = 1 << 1,
};

inline constexpr PropertyTag operator|(PropertyTag a, PropertyTag b) {
    return static_cast<PropertyTag>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
inline constexpr PropertyTag operator&(PropertyTag a, PropertyTag b) {
    return static_cast<PropertyTag>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}
inline constexpr bool has_tag(PropertyTag tags, PropertyTag flag) {
    return (tags & flag) != PropertyTag::None;
}

/// Called after the user edits a value. Receives the proposed new value,
/// may clamp/modify it in place. Return false to reject the edit entirely.
using ValidateFn = std::function<bool(std::any& value)>;

struct PropertyDescriptor {
    std::string name;
    std::string label;
    std::any value;
    PropertyTag tags{PropertyTag::None};
    float drag_speed{0.01f};
    float min_val{0.0f};
    float max_val{0.0f};
    ValidateFn validate;  // optional
};

}  // namespace pts::rendering
