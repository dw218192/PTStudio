#include "include/shader.h"
#include "include/glTexture.h"
#include <fstream>

#include "enumIter.h"

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

    auto link_shaders(GLuint program, View<EArray<ShaderType, ShaderRef>> shaders) noexcept -> tl::expected<void, std::string> {
        for (auto&& shader : shaders) {
            glAttachShader(program, shader->handle());
            CHECK_GL_ERROR();
        }

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

        for (auto&& shader : shaders) {
            glDetachShader(program, shader->handle());
            CHECK_GL_ERROR();
        }

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

auto Shader::get_type() const noexcept -> ShaderType {
    return to_shader_type(m_type);
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

auto Shader::clone(ViewPtr<Shader> other) noexcept -> tl::expected<ShaderRef, std::string> {
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

ShaderProgram::ShaderProgram(EArray<ShaderType, ShaderRef> shaders, GLuint handle) noexcept
    : GLResource{ handle } 
{
    m_shaders = std::move(shaders);
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept {
    swap(std::move(other));
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    swap(std::move(other));
    return *this;
}

auto ShaderProgram::set_texture(std::string_view name, ViewPtr<GLTexture> tex, GLuint slot) const noexcept -> tl::expected<void, std::string> {
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

auto ShaderProgram::try_recompile(tcb::span<StageDesc<std::string_view> const> new_srcs) noexcept -> tl::expected<void, std::string> {
    ShaderProgramRef copy;
    TL_ASSIGN(copy, ShaderProgram::clone(this));
    TL_CHECK_FWD(copy->recompile(new_srcs));
    swap(std::move(*copy));

	return {};
}

auto ShaderProgram::recompile(tcb::span<StageDesc<std::string_view> const> new_srcs) noexcept -> tl::expected<void, std::string> {
    if (!m_handle) {
        return TL_ERROR( "Shader program is not valid" );
    }

    for (auto&& [type, src] : new_srcs) {
        if (!m_shaders[type]) {
            TL_ASSIGN(m_shaders[type], Shader::from_src(type, src));
        } else {
            TL_CHECK_FWD(m_shaders[type]->recompile(src));
        }
    }
	TL_CHECK_FWD(link_shaders(m_handle, m_shaders));

    return {};
}

auto ShaderProgram::get_stage(ShaderType type) const noexcept -> ViewPtr<Shader> {
    return m_shaders[type].get();
}

void ShaderProgram::swap(ShaderProgram&& other) noexcept {
    m_shaders.swap(other.m_shaders);
    this->GLResource::swap(std::move(other));
}

auto ShaderProgram::from_shaders(tcb::span<StageDesc<ShaderRef>> shaders) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    auto const program = glCreateProgram();
    if (!program) {
        return TL_ERROR( "Failed to create shader program" );
    }

    EArray<ShaderType, ShaderRef> data;
	for (auto&& [type, shader] : shaders) {
        data[type] = std::move(shader);
    }
    TL_CHECK(link_shaders(program, data));

    return ShaderProgramRef{ new ShaderProgram { std::move(data), program}, GLResourceDeleter{}};
}

auto ShaderProgram::clone(ViewPtr<ShaderProgram> other) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    if (!other || !other->valid()) {
        return TL_ERROR("Invalid shader program");
    }

    std::vector<StageDesc<ShaderRef>> stages;
    for(auto type : EIter<ShaderType>{}) {
        ShaderRef shader;
        TL_ASSIGN(shader, Shader::clone(other->m_shaders[type].get()));
        stages.emplace_back(type, std::move(shader));
    }
    return from_shaders(stages);
}

auto ShaderProgram::from_files(tcb::span<StageDesc<std::string_view> const> files) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    std::vector<StageDesc<ShaderRef>> stages;
    for (auto&& [type, src] : files) {
        ShaderRef shader;
        TL_ASSIGN(shader, Shader::from_file(type, src));
        stages.emplace_back(type, std::move(shader));
    }

    return from_shaders(stages);
}

auto ShaderProgram::from_srcs(tcb::span<StageDesc<std::string_view> const> srcs) noexcept -> tl::expected<ShaderProgramRef, std::string> {
    std::vector<StageDesc<ShaderRef>> stages;
    for (auto&& [type, src] : srcs) {
        ShaderRef shader;
        TL_ASSIGN(shader, Shader::from_src(type, src));
        stages.emplace_back(type, std::move(shader));
    }

    return from_shaders(stages);
}