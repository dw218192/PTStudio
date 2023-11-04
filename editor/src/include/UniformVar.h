#pragma once
#include <string>
#include <GL/glew.h>
#include <variant>
#include <glm/glm.hpp>
#include <tcb/span.hpp>
#include "utils.h"

enum ShaderVariableType {
    Mat3 = 1,
    Mat4 = 2,
    Vec2 = 1 << 2,
    Vec3 = 1 << 3,
    Vec4 = 1 << 4,
    Float = 1 << 5,
    Int = 1 << 6,
    Sampler2D = 1 << 7,
    IVec2 = 1 << 8,
    IVec3 = 1 << 9,
    IVec4 = 1 << 10,
    Vec3Array = 1 << 11,
    Vec4Array = 1 << 12,
    Mat4Array = 1 << 14,
    FloatArray = 1 << 15,
    IntArray = 1 << 16,
};

template<typename T>
constexpr auto type_to_enum_msk(T) {
    if constexpr (std::is_same_v<T, glm::vec3>) {
        return ShaderVariableType::Vec3;
    } else if constexpr (std::is_same_v<T, glm::vec2>) {
        return ShaderVariableType::Vec2;
    } else if constexpr (std::is_same_v<T, glm::vec4>) {
        return ShaderVariableType::Vec4;
    } else if constexpr (std::is_same_v<T, glm::mat3>) {
        return ShaderVariableType::Mat3;
    } else if constexpr (std::is_same_v<T, glm::mat4>) {
        return ShaderVariableType::Mat4;
    } else if constexpr (std::is_same_v<T, float>) {
        return ShaderVariableType::Float;
    } else if constexpr (std::is_same_v<T, int>) {
        return ShaderVariableType::Int | ShaderVariableType::Sampler2D;
    } else if constexpr (std::is_same_v<T, glm::ivec2>) {
        return ShaderVariableType::IVec2;
    } else if constexpr (std::is_same_v<T, glm::ivec3>) {
        return ShaderVariableType::IVec3;
    } else if constexpr (std::is_same_v<T, glm::ivec4>) {
        return ShaderVariableType::IVec4;
    } else if constexpr (std::is_same_v<T, tcb::span<glm::vec3 const>>) {
        return ShaderVariableType::Vec3Array;
    } else if constexpr (std::is_same_v<T, tcb::span<glm::vec4 const>>) {
        return ShaderVariableType::Vec4Array;
    } else if constexpr (std::is_same_v<T, tcb::span<glm::mat4 const>>) {
        return ShaderVariableType::Mat4Array;
    } else if constexpr (std::is_same_v<T, tcb::span<float const>>) {
        return ShaderVariableType::FloatArray;
    } else if constexpr (std::is_same_v<T, tcb::span<int const>>) {
        return ShaderVariableType::IntArray;
    } else {
        static_assert(false, "Invalid type");
    }
}

constexpr auto get_type_str(ShaderVariableType type) {
    switch (type) {
    case ShaderVariableType::Mat3:
        return "mat3";
    case ShaderVariableType::Mat4:
        return "mat4";
    case ShaderVariableType::Vec2:
        return "vec2";
    case ShaderVariableType::Vec3:
        return "vec3";
    case ShaderVariableType::Vec4:
        return "vec4";
    case ShaderVariableType::Float:
        return "float";
    case ShaderVariableType::Int:
        return "int";
    case ShaderVariableType::Sampler2D:
        return "sampler2D";
    case ShaderVariableType::IVec2:
        return "ivec2";
    case ShaderVariableType::IVec3:
        return "ivec3";
    case ShaderVariableType::IVec4:
        return "ivec4";
    case ShaderVariableType::Vec3Array:
        return "vec3[]";
    case ShaderVariableType::Vec4Array:
        return "vec4[]";
    case ShaderVariableType::Mat4Array:
        return "mat4[]";
    case ShaderVariableType::FloatArray:
        return "float[]";
    case ShaderVariableType::IntArray:
        return "int[]";
    default:
        return "unknown";
    }
}

struct UniformVar {
    friend struct ShaderProgram;

    static auto create(GLenum type, GLint loc, std::string_view name)  noexcept -> tl::expected<UniformVar, std::string>;

    UniformVar() = default;

    template<typename T>
    UniformVar(ShaderVariableType type, GLint loc, T value) noexcept
        : m_type(type), m_loc(loc), value(std::move(value)) {}
    
    template<typename T>
    void set_value(T value) noexcept {
        this->value = std::move(value);
    }
    template<typename T>
    auto get_value() const noexcept -> View<T> {
        return std::get<T>(value);
    }
    auto get_type() const noexcept -> ShaderVariableType {
        return m_type;
    }
    auto get_loc() const noexcept -> GLint {
        return m_loc;
    }
private:
    auto upload() noexcept -> tl::expected<void, std::string>;

    ShaderVariableType m_type;
    GLint m_loc;
    std::variant<
        glm::mat3,
        glm::mat4,
        glm::vec2,
        glm::vec3,
        glm::vec4,
        glm::ivec2,
        glm::ivec3,
        glm::ivec4,
        float,
        int,
        tcb::span<glm::vec3 const>,
        tcb::span<glm::vec4 const>,
        tcb::span<glm::mat4 const>,
        tcb::span<float const>,
        tcb::span<int const>
    > value;
};
