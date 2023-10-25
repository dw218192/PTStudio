#include "include/UniformVar.h"

#include <glm/gtc/type_ptr.hpp>

auto to_variable_type(GLenum type) noexcept -> ShaderVariableType {
    switch (type) {
    case GL_FLOAT_MAT4:
        return ShaderVariableType::Mat4;
    case GL_FLOAT_MAT3:
        return ShaderVariableType::Mat3;
    case GL_FLOAT_VEC2:
        return ShaderVariableType::Vec2;
    case GL_FLOAT_VEC3:
        return ShaderVariableType::Vec3;
    case GL_FLOAT_VEC4:
        return ShaderVariableType::Vec4;
    case GL_FLOAT:
        return ShaderVariableType::Float;
    case GL_INT:
        return ShaderVariableType::Int;
    case GL_SAMPLER_2D:
        return ShaderVariableType::Sampler2D;
    default:
        return ShaderVariableType::Int;
    }
}

auto UniformVar::create(GLenum gltype, GLint loc) noexcept -> tl::expected<UniformVar, std::string> {
	switch (auto const type = to_variable_type(gltype)) {
    case ShaderVariableType::Mat3:
        return UniformVar{ type, loc, glm::mat3{} };
    case ShaderVariableType::Mat4:
        return UniformVar{ type, loc, glm::mat4{} };
    case ShaderVariableType::Vec2:
        return UniformVar{ type, loc, glm::vec2{} };
    case ShaderVariableType::Vec3:
        return UniformVar{ type, loc, glm::vec3{} };
    case ShaderVariableType::Vec4:
        return UniformVar{ type, loc, glm::vec4{} };
    case ShaderVariableType::Float:
        return UniformVar{ type, loc, 0.0f };
    case ShaderVariableType::Int:
    case ShaderVariableType::Sampler2D:
        return UniformVar{ type, loc, 0 };
    default:
        return TL_ERROR( "Unsupported shader variable type" );
    }
}

auto UniformVar::upload() noexcept -> tl::expected <void, std::string> {
    switch (get_type()) {
    case ShaderVariableType::Mat3:
        glUniformMatrix3fv(get_loc(), 1, GL_FALSE, glm::value_ptr(std::get<glm::mat3>(value)));
        break;
    case ShaderVariableType::Mat4:
        glUniformMatrix4fv(get_loc(), 1, GL_FALSE, glm::value_ptr(std::get<glm::mat4>(value)));
        break;
    case ShaderVariableType::Vec2:
        glUniform2fv(get_loc(), 1, glm::value_ptr(std::get<glm::vec2>(value)));
        break;
    case ShaderVariableType::Vec3:
        glUniform3fv(get_loc(), 1, glm::value_ptr(std::get<glm::vec3>(value)));
        break;
    case ShaderVariableType::Vec4:
        glUniform4fv(get_loc(), 1, glm::value_ptr(std::get<glm::vec4>(value)));
        break;
    case ShaderVariableType::Float:
        glUniform1f(get_loc(), std::get<float>(value));
        break;
    case ShaderVariableType::Int:
        glUniform1i(get_loc(), std::get<int>(value));
        break;
    case ShaderVariableType::Sampler2D:
        glUniform1i(get_loc(), std::get<int>(value));
        break;
    default:
        return TL_ERROR("Unsupported shader variable type");
    }

    CHECK_GL_ERROR();
    return {};
}
