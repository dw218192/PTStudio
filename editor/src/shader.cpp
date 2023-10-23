#include "include/shader.h"
#include "include/glTexture.h"
#include <fstream>

namespace {
    auto to_gl_type(ShaderType type) noexcept -> GLenum {
        switch (type) {
        case ShaderType::Vertex:
            return GL_VERTEX_SHADER;
        case ShaderType::Fragment:
            return GL_FRAGMENT_SHADER;
        }
        return 0;
    }
    auto to_shader_type(GLenum type) noexcept -> ShaderType {
        switch (type) {
        case GL_VERTEX_SHADER:
            return ShaderType::Vertex;
        case GL_FRAGMENT_SHADER:
            return ShaderType::Fragment;
        }
        return ShaderType::Vertex;
    }
    auto compile_shader(GLuint handle, std::string_view src) noexcept -> tl::expected<void, std::string> {
        auto src_ptr = src.data();
        glShaderSource(handle, 1, &src_ptr, nullptr);
        glCompileShader(handle);

        GLint status;
        glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
        if (status == GL_FALSE) {
            GLint maxLength = 0;
            glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &maxLength);

            std::string infoLog(maxLength, '\0');
            glGetShaderInfoLog(handle, maxLength, &maxLength, infoLog.data());

            infoLog = "Failed to compile shader:\n" + infoLog;
            return TL_ERROR( infoLog );
        }
        return {};
    }

    auto create_shader(GLenum type, std::string_view src) noexcept -> tl::expected<GLuint, std::string> {
        auto handle = glCreateShader(type);
        if (!handle) {
            if (type == GL_VERTEX_SHADER) {
                return TL_ERROR( "Failed to create ertex shader" );
            } else {
                return TL_ERROR( "Failed to create pixel shader" );
            }
        }
        TL_CHECK(compile_shader(handle, src));
        return handle;
    }

    auto link_shaders(GLuint program, GLuint vs, GLuint ps) noexcept -> tl::expected<void, std::string> {
        glAttachShader(program, vs);
        CHECK_GL_ERROR();

        glAttachShader(program, ps);
        CHECK_GL_ERROR();

        glLinkProgram(program);
        CHECK_GL_ERROR();

        GLint status;
        glGetProgramiv(program, GL_LINK_STATUS, &status);
        CHECK_GL_ERROR();

        if (status == GL_FALSE) {
            GLint maxLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
            CHECK_GL_ERROR();

            std::string infoLog(maxLength, '\0');
            glGetProgramInfoLog(program, maxLength, &maxLength, infoLog.data());
            CHECK_GL_ERROR();

            infoLog = "Failed to compile shader:\n" + infoLog;
            return TL_ERROR( infoLog );
        }
        glDetachShader(program, vs);
        CHECK_GL_ERROR();

        glDetachShader(program, ps);
        CHECK_GL_ERROR();

        return {};
    }
};

Shader::Shader(GLenum type, GLuint handle, std::string_view src) noexcept 
    : GLResource(handle), m_type(type), m_src(src) {}

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

auto Shader::recompile(std::string_view new_src) -> tl::expected<void, std::string> {
    if (!m_handle) {
        return TL_ERROR( "Shader is not valid" );
    }
    TL_CHECK(compile_shader(m_handle, new_src));
    m_src = new_src;
    return {};
}

void Shader::swap(Shader&& other) noexcept {
    m_src.swap(other.m_src);
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

auto Shader::clone(Shader const* other) noexcept -> tl::expected<ShaderRef, std::string> {
    if (!other || !other->valid()) {
        return TL_ERROR("Invalid shader");
    }
    return from_src(to_shader_type(other->m_type), other->m_src);
}

auto Shader::from_src(ShaderType type, std::string_view src) noexcept -> tl::expected<ShaderRef, std::string> {
    auto gltype = to_gl_type(type);
    GLuint handle;
    TL_ASSIGN(handle, create_shader(gltype, src));
    auto ret = ShaderRef{ new Shader { gltype, handle, src }, GLResourceDeleter{ } };
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

auto ShaderProgram::set_texture(std::string_view name, GLTexture const* tex, GLuint slot) const noexcept -> tl::expected<void, std::string> {
    if (!valid()) {
        return TL_ERROR( "Invalid shader program" );
    }

    glActiveTexture(GL_TEXTURE0 + slot);
    TL_CHECK_FWD(tex->bind());
    TL_CHECK_FWD(set_uniform(name, static_cast<int>(slot)));

    return {};
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

auto ShaderProgram::recompile(std::string_view new_vs, std::string_view new_ps) noexcept -> tl::expected<void, std::string> {
    if (!m_handle) {
        return TL_ERROR( "Shader program is not valid" );
    }

    TL_CHECK_FWD(m_vs->recompile(new_vs));
    TL_CHECK_FWD(m_ps->recompile(new_ps));
    TL_CHECK_FWD(link_shaders(m_handle, m_vs->handle(), m_ps->handle()));

    return {};
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
    TL_CHECK(link_shaders(program, vs->handle(), ps->handle()));
    return ShaderProgramRef{ new ShaderProgram { std::move(vs), std::move(ps), program }, GLResourceDeleter{ } };
}

auto ShaderProgram::clone(ShaderProgram const* other) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    if (!other || !other->valid()) {
        return TL_ERROR("Invalid shader program");
    }
    ShaderRef vs, ps;
    TL_ASSIGN(vs, Shader::clone(other->m_vs.get()));
    TL_ASSIGN(ps, Shader::clone(other->m_ps.get()));
    
    return from_shaders(std::move(vs), std::move(ps));
}

auto ShaderProgram::from_files(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    ShaderRef vs_shader, ps_shader;

    TL_ASSIGN(vs_shader, Shader::from_file(ShaderType::Vertex, vs));
    TL_ASSIGN(ps_shader, Shader::from_file(ShaderType::Fragment, ps));

    return from_shaders(std::move(vs_shader), std::move(ps_shader));
}

auto ShaderProgram::from_srcs(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    ShaderRef vs_shader, ps_shader;

	TL_ASSIGN(vs_shader, Shader::from_src(ShaderType::Vertex, vs));
    TL_ASSIGN(ps_shader, Shader::from_src(ShaderType::Fragment, ps));

    return from_shaders(std::move(vs_shader), std::move(ps_shader));
}