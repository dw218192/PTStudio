#pragma once
#include <any>
#include <cstdint>
#include <string>
#include <vector>

namespace pts::rendering {

enum class PropertyTag : uint8_t {
    None = 0,
    Color = 1 << 0,
    ReadOnly = 1 << 1,
};

inline PropertyTag operator|(PropertyTag a, PropertyTag b) {
    return static_cast<PropertyTag>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
inline PropertyTag operator&(PropertyTag a, PropertyTag b) {
    return static_cast<PropertyTag>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}
inline bool has_tag(PropertyTag tags, PropertyTag flag) {
    return (tags & flag) != PropertyTag::None;
}

struct PropertyDescriptor {
    std::string name;
    std::string label;
    std::any value;
    PropertyTag tags{PropertyTag::None};
    float drag_speed{0.01f};
    float min_val{0.0f};
    float max_val{0.0f};
};

}  // namespace pts::rendering
