#include "include/shader.h"
#include <fstream>

Shader::Shader(GLenum type, GLuint handle) noexcept : GLResource(handle), m_type(type) {}

Shader::~Shader() noexcept {
    if (m_handle) {
        glDeleteShader(m_handle);
    }
}

Shader::Shader(Shader&& other) noexcept {
    swap(std::move(other));
}

Shader& Shader::operator=(Shader&& other) noexcept {
    swap(std::move(other));
    return *this;
}

void Shader::swap(Shader&& other) noexcept {
    m_type = other.m_type;
    other.m_type = 0;
    this->GLResource::swap(std::move(other));
}

auto Shader::from_file(ShaderType type, std::string_view file) noexcept -> tl::expected<ShaderRef, std::string> {
    std::ifstream stream(file.data());
    if (!stream.is_open()) {
        return TL_ERROR( "Failed to open vetex shader file" );
    }

    std::string src{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};    
    return from_src(type, src);
}

auto Shader::from_src(ShaderType type, std::string_view src) noexcept -> tl::expected<ShaderRef, std::string> {
    GLenum gltype {0};
	switch (type) {
    case ShaderType::Vertex:
        gltype = GL_VERTEX_SHADER;
        break;
    case ShaderType::Fragment:
        gltype = GL_FRAGMENT_SHADER;
        break;
    }

	GLuint handle = glCreateShader(gltype);
    if (!handle) {
        if (gltype == GL_VERTEX_SHADER) {
            return TL_ERROR( "Failed to create ertex shader" );
        } else {
            return TL_ERROR( "Failed to create pixel shader" );
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
    auto ret = ShaderRef{ new Shader { gltype, handle }, GLResourceDeleter{ } };
	return ret;
}




ShaderProgram::~ShaderProgram() noexcept {
    if (m_handle) {
        glDeleteProgram(m_handle);
    }
}

ShaderProgram::ShaderProgram(ShaderRef vs, ShaderRef ps, GLuint handle) noexcept
    : GLResource{ handle }, m_vs{ std::move(vs) }, m_ps{ std::move(ps) } { }

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept {
    swap(std::move(other));
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    swap(std::move(other));
    return *this;
}

auto ShaderProgram::bind() const noexcept -> tl::expected<void, std::string> {
    if (!m_handle) {
        return TL_ERROR("shader is not valid");
    }
    glUseProgram(m_handle);
    CHECK_GL_ERROR();
    return {};
}

void ShaderProgram::unbind() noexcept {
    glUseProgram(0);
}

void ShaderProgram::swap(ShaderProgram&& other) noexcept {
    m_ps = std::move(other.m_ps);
    m_vs = std::move(other.m_vs);
    this->GLResource::swap(std::move(other));
}

auto ShaderProgram::from_shaders(ShaderRef vs, ShaderRef ps) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    if(!vs->valid()) {
        return TL_ERROR( "Invalid vertex shder" );
    }
    if(!ps->valid()) {
        return TL_ERROR( "Invalid pixel shaer" );
    }

    auto const program = glCreateProgram();
    if (!program) {
        return TL_ERROR( "Failed to create hader program" );
    }
    glAttachShader(program, vs->handle());
    glAttachShader(program, ps->handle());
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
    return ShaderProgramRef{ new ShaderProgram { std::move(vs), std::move(ps), program }, GLResourceDeleter{ } };
}

auto ShaderProgram::from_files(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgramRef, std::string> {
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

auto ShaderProgram::from_srcs(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgramRef, std::string> {
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