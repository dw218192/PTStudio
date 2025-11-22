#include "include/uniformVar.h"

#include <glm/gtc/type_ptr.hpp>

using namespace PTS;

auto to_variable_type(GLenum type, std::string_view name) noexcept
    -> tl::expected<ShaderVariableType, std::string> {
    if (name.find('[') != std::string_view::npos) {
        switch (type) {
            case GL_FLOAT_VEC3:
                return ShaderVariableType::Vec3Array;
            case GL_FLOAT_VEC4:
                return ShaderVariableType::Vec4Array;
            case GL_FLOAT_MAT4:
                return ShaderVariableType::Mat4Array;
            case GL_FLOAT:
                return ShaderVariableType::FloatArray;
            case GL_INT:
                return ShaderVariableType::IntArray;
            default:
                return TL_ERROR("Unsupported shader variable type");
        }
    } else {
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
            case GL_INT_VEC2:
                return ShaderVariableType::IVec2;
            case GL_INT_VEC3:
                return ShaderVariableType::IVec3;
            case GL_INT_VEC4:
                return ShaderVariableType::IVec4;
            default:
                return TL_ERROR("Unsupported shader variable type");
        }
    }
}

auto UniformVar::create(GLenum gltype, GLint loc, std::string_view name) noexcept
    -> tl::expected<UniformVar, std::string> {
    ShaderVariableType type;
    TL_TRY_ASSIGN(type, to_variable_type(gltype, name));
    switch (type) {
        case ShaderVariableType::Mat3:
            return UniformVar{type, loc, glm::mat3{}};
        case ShaderVariableType::Mat4:
            return UniformVar{type, loc, glm::mat4{}};
        case ShaderVariableType::Vec2:
            return UniformVar{type, loc, glm::vec2{}};
        case ShaderVariableType::Vec3:
            return UniformVar{type, loc, glm::vec3{}};
        case ShaderVariableType::Vec4:
            return UniformVar{type, loc, glm::vec4{}};
        case ShaderVariableType::Float:
            return UniformVar{type, loc, 0.0f};
        case ShaderVariableType::Int:
        case ShaderVariableType::Sampler2D:
            return UniformVar{type, loc, 0};
        case ShaderVariableType::IVec2:
            return UniformVar{type, loc, glm::ivec2{}};
        case ShaderVariableType::IVec3:
            return UniformVar{type, loc, glm::ivec3{}};
        case ShaderVariableType::IVec4:
            return UniformVar{type, loc, glm::ivec4{}};
        case ShaderVariableType::Vec3Array:
            return UniformVar{type, loc, std::vector<glm::vec3>{}};
        case ShaderVariableType::Vec4Array:
            return UniformVar{type, loc, std::vector<glm::vec4>{}};
        case ShaderVariableType::Mat4Array:
            return UniformVar{type, loc, std::vector<glm::mat4>{}};
        case ShaderVariableType::FloatArray:
            return UniformVar{type, loc, std::vector<float>{}};
        case ShaderVariableType::IntArray:
            return UniformVar{type, loc, std::vector<int>{}};
        default:
            return TL_ERROR("Unsupported shader variable type");
    }
}

auto UniformVar::upload() noexcept -> tl::expected<void, std::string> {
    if (m_loc == -1) {
        return TL_ERROR("Invalid uniform location");
    }
    switch (get_type()) {
        case ShaderVariableType::Mat3:
            glUniformMatrix3fv(m_loc, 1, GL_FALSE, glm::value_ptr(std::get<glm::mat3>(value)));
            break;
        case ShaderVariableType::Mat4:
            glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm::value_ptr(std::get<glm::mat4>(value)));
            break;
        case ShaderVariableType::Vec2:
            glUniform2fv(m_loc, 1, glm::value_ptr(std::get<glm::vec2>(value)));
            break;
        case ShaderVariableType::Vec3:
            glUniform3fv(m_loc, 1, glm::value_ptr(std::get<glm::vec3>(value)));
            break;
        case ShaderVariableType::Vec4:
            glUniform4fv(m_loc, 1, glm::value_ptr(std::get<glm::vec4>(value)));
            break;
        case ShaderVariableType::Float:
            glUniform1f(m_loc, std::get<float>(value));
            break;
        case ShaderVariableType::Int:
        case ShaderVariableType::Sampler2D:
            glUniform1i(m_loc, std::get<int>(value));
            break;
        case ShaderVariableType::IVec2:
            glUniform2iv(m_loc, 1, glm::value_ptr(std::get<glm::ivec2>(value)));
            break;
        case ShaderVariableType::IVec3:
            glUniform3iv(m_loc, 1, glm::value_ptr(std::get<glm::ivec3>(value)));
            break;
        case ShaderVariableType::IVec4:
            glUniform4iv(m_loc, 1, glm::value_ptr(std::get<glm::ivec4>(value)));
            break;
        case ShaderVariableType::Vec3Array: {
            using type = tcb::span<glm::vec3 const>;
            if (std::get<type>(value).size()) {
                glUniform3fv(m_loc, std::get<type>(value).size(),
                             glm::value_ptr(std::get<type>(value)[0]));
            }
        } break;
        case ShaderVariableType::Vec4Array: {
            using type = tcb::span<glm::vec4 const>;
            if (std::get<type>(value).size()) {
                glUniform4fv(m_loc, std::get<type>(value).size(),
                             glm::value_ptr(std::get<type>(value)[0]));
            }
        } break;
        case ShaderVariableType::Mat4Array: {
            using type = tcb::span<glm::mat4 const>;
            if (std::get<type>(value).size()) {
                glUniformMatrix4fv(m_loc, std::get<type>(value).size(), GL_FALSE,
                                   glm::value_ptr(std::get<type>(value)[0]));
            }
        } break;
        case ShaderVariableType::FloatArray: {
            using type = tcb::span<float const>;
            if (std::get<type>(value).size()) {
                glUniform1fv(m_loc, std::get<type>(value).size(), std::get<type>(value).data());
            }
        } break;
        case ShaderVariableType::IntArray: {
            using type = tcb::span<int const>;
            if (std::get<type>(value).size()) {
                glUniform1iv(m_loc, std::get<type>(value).size(), std::get<type>(value).data());
            }
        } break;
        default:
            return TL_ERROR("Unsupported shader variable type");
    }

    CHECK_GL_ERROR();
    return {};
}
