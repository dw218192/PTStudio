#include "include/shader.h"

#include <fstream>
#include <glm/gtc/type_ptr.hpp>

Shader::Shader(ShaderType type) noexcept : m_handle(0) {
    switch (type) {
    case ShaderType::Vertex:
        m_type = GL_VERTEX_SHADER;
        break;
    case ShaderType::Fragment:
        m_type = GL_FRAGMENT_SHADER;
        break;
    }
}

Shader::~Shader() noexcept {
    if (m_handle) {
        glDeleteShader(m_handle);
    }
}

Shader::Shader(Shader&& other) noexcept : m_type{other.m_type}, m_handle{other.m_handle} {
    other.m_handle = 0;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    m_type = other.m_type;
    m_handle = other.m_handle;
    other.m_handle = 0;
    return *this;
}

auto Shader::from_file(ShaderType type, std::string_view file) noexcept -> tl::expected<Shader, std::string> {
    std::ifstream stream(file.data());
    if (!stream.is_open()) {
        return tl::unexpected{ "Failed to open vertex shader file" };
    }

    std::string src{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};    
    return from_src(type, src);
}

auto Shader::from_src(ShaderType type, std::string_view src) noexcept -> tl::expected<Shader, std::string> {
    Shader ret{ type };
    GLuint handle = glCreateShader(ret.m_type);
    if (!handle) {
        if (ret.m_type == GL_VERTEX_SHADER) {
            return tl::unexpected{ "Failed to create vertex shader" };
        } else {
            return tl::unexpected{ "Failed to create pixel shader" };
        }
    }

    auto src_ptr = src.data();
    glShaderSource(handle, 1, &src_ptr, nullptr);
    glCompileShader(handle);

    GLint status;
    glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
    if(status == GL_FALSE) {
        GLint maxLength = 0;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &maxLength);

        std::string infoLog(maxLength, '\0');
        glGetShaderInfoLog(handle, maxLength, &maxLength, infoLog.data());

        infoLog = "Failed to compile shader:\n" + infoLog;
        return tl::unexpected{ infoLog };
    }

    ret.m_handle = handle;
    return ret;
}

ShaderProgram::~ShaderProgram() noexcept {
    if (m_handle) {
        glDeleteProgram(m_handle);
    }
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
	: m_handle{other.m_handle}, m_vs{std::move(other.m_vs)}, m_ps{std::move(other.m_ps)}
{
    other.m_handle = 0;
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    m_handle = other.m_handle;
    m_vs = std::move(other.m_vs);
    m_ps = std::move(other.m_ps);
    other.m_handle = 0;
    return *this;
}

void ShaderProgram::use() const noexcept {
    if (m_handle) {
        glUseProgram(m_handle);
    }
}
void ShaderProgram::unuse() const noexcept {
    glUseProgram(0);
}

auto ShaderProgram::from_shaders(Shader&& vs, Shader&& ps) noexcept -> tl::expected<ShaderProgram, std::string> {
    if(!vs.valid()) {
        return tl::unexpected{ "Invalid vertex shader" };
    }
    if(!ps.valid()) {
        return tl::unexpected{ "Invalid pixel shader" };
    }

    GLuint program = glCreateProgram();
    if (!program) {
        return tl::unexpected{ "Failed to create shader program" };
    }
    glAttachShader(program, vs.m_handle);
    glAttachShader(program, ps.m_handle);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if(status == GL_FALSE) {
        GLint maxLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

        std::string infoLog(maxLength, '\0');
        glGetProgramInfoLog(program, maxLength, &maxLength, infoLog.data());

        infoLog = "Failed to compile shader:\n" + infoLog;
        return tl::unexpected{ infoLog };
    }
    ShaderProgram ret{ std::move(vs), std::move(ps) };
    ret.m_handle = program;
    return ret;
}

auto ShaderProgram::set_uniform(std::string_view name, glm::mat4 const& value) const noexcept -> tl::expected<void, std::string> {
    GLint const location = glGetUniformLocation(m_handle, name.data());
    if (location == -1) {
        return tl::unexpected{ "Failed to find uniform location" };
    }
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(value));
    auto const err = glGetError();
    if (err != GL_NO_ERROR) {
        return tl::unexpected{ reinterpret_cast<char const*>(glewGetErrorString(err))};
    }
    return {};
}

auto ShaderProgram::set_uniform(std::string_view name, glm::vec3 const& value) const noexcept -> tl::expected<void, std::string> {
    GLint const location = glGetUniformLocation(m_handle, name.data());
    if (location == -1) {
        return tl::unexpected{ "Failed to find uniform location" };
    }
    glUniform3fv(location, 1, glm::value_ptr(value));
    auto const err = glGetError();
    if (err != GL_NO_ERROR) {
        return tl::unexpected{ reinterpret_cast<char const*>(glewGetErrorString(err)) };
    }

    return {};
}
