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
        return TL_ERROR( "Failed to open vetex shader file" );
    }

    std::string src{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};    
    return from_src(type, src);
}

auto Shader::from_src(ShaderType type, std::string_view src) noexcept -> tl::expected<Shader, std::string> {
    Shader ret{ type };
    GLuint handle = glCreateShader(ret.m_type);
    if (!handle) {
        if (ret.m_type == GL_VERTEX_SHADER) {
            return TL_ERROR( "Failed to create ertex shader" );
        } else {
            return TL_ERROR( "Failed to create ixel shader" );
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
        return TL_ERROR( infoLog );
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
        return TL_ERROR( "Invalid vertex shder" );
    }
    if(!ps.valid()) {
        return TL_ERROR( "Invalid pixel shaer" );
    }

    GLuint program = glCreateProgram();
    if (!program) {
        return TL_ERROR( "Failed to create hader program" );
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
        return TL_ERROR( infoLog );
    }
  ShaderProgram ret{ std::move(vs), std::move(ps) };
    ret.m_handle = program;
    return ret;
}

auto ShaderProgram::from_files(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgram, std::string> {
    auto vs_shader = Shader::from_file(ShaderType::Vertex, vs);
    if (!vs_shader) {
        return TL_ERROR( vs_shader.error() );
    }
    auto ps_shader = Shader::from_file(ShaderType::Fragment, ps);
    if (!ps_shader) {
        return TL_ERROR( ps_shader.error() );
    }
    return from_shaders(std::move(vs_shader.value()), std::move(ps_shader.value()));
}

auto ShaderProgram::from_srcs(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgram, std::string> {
    auto vs_shader = Shader::from_src(ShaderType::Vertex, vs);
    if (!vs_shader) {
        return TL_ERROR( vs_shader.error() );
    }
    auto ps_shader = Shader::from_src(ShaderType::Fragment, ps);
    if (!ps_shader) {
        return TL_ERROR( ps_shader.error() );
    }
    return from_shaders(std::move(vs_shader.value()), std::move(ps_shader.value()));
}