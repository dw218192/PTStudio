#pragma once
#include <string>
#include <GL/glew.h>
#include <variant>
#include <glm/glm.hpp>
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
    } else {
        static_assert(false, "Unsupported type");
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
    default:
        return "unknown";
    }
}

struct UniformVar {
    friend struct ShaderProgram;

    static auto create(GLenum type, GLint loc)  noexcept -> tl::expected<UniformVar, std::string>;

    UniformVar() = default;

    template<typename T>
    UniformVar(ShaderVariableType type, GLint loc, T value) noexcept
        : type(type), loc(loc), value(std::move(value)) {}
    
    template<typename T>
    void set_value(T value) noexcept {
        this->value = std::move(value);
    }
    template<typename T>
    auto get_value() const noexcept -> View<T> {
        return std::get<T>(value);
    }
    auto get_type() const noexcept -> ShaderVariableType {
        return type;
    }
    auto get_loc() const noexcept -> GLint {
        return loc;
    }
private:
    auto upload() noexcept -> tl::expected<void, std::string>;

    ShaderVariableType type;
    GLint loc;
    std::variant<
        glm::mat3,
        glm::mat4,
        glm::vec2,
        glm::vec3,
        glm::vec4,
        float,
        int
    > value;
};
