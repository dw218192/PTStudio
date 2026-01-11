#include "shader.h"

#include <core/enumIter.h>
#include <core/transactionScope.h>

#include <fstream>

#include "glTexture.h"

using namespace PTS;

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

auto compile_shader(GLuint handle,
                    std::string_view src) noexcept -> tl::expected<void, std::string> {
    auto const src_ptr = src.data();
    glShaderSource(handle, 1, &src_ptr, nullptr);
    glCompileShader(handle);

    GLint status;
    glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint maxLength = 0;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &maxLength);

        std::string infoLog(maxLength, '\0');
        glGetShaderInfoLog(handle, maxLength, &maxLength, infoLog.data());
        return TL_ERROR("Failed to compile shader \n {} \n {}", infoLog, src);
    }
    return {};
}

auto create_shader(GLenum type,
                   std::string_view src) noexcept -> tl::expected<GLuint, std::string> {
    GLuint handle{0};
    TransactionScope transaction{[handle]() {
        if (handle) {
            glDeleteShader(handle);
        }
    }};

    handle = glCreateShader(type);
    if (!handle) {
        if (type == GL_VERTEX_SHADER) {
            return TL_ERROR("Failed to create ertex shader");
        } else {
            return TL_ERROR("Failed to create pixel shader");
        }
    }
    TL_CHECK(compile_shader(handle, src));

    transaction.commit();
    return handle;
}

auto link_shaders(GLuint program, View<EArray<ShaderType, ShaderRef>> shaders) noexcept
    -> tl::expected<void, std::string> {
    for (auto&& shader : shaders.get()) {
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
        return TL_ERROR(infoLog);
    }

    for (auto&& shader : shaders.get()) {
        glDetachShader(program, shader->handle());
        CHECK_GL_ERROR();
    }

    return {};
}

auto get_uniforms(GLuint program) noexcept -> tl::expected<ShaderProgram::UniformMap, std::string> {
    ShaderProgram::UniformMap uniforms;
    GLint count;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &count);
    CHECK_GL_ERROR();

    for (GLint i = 0; i < count; ++i) {
        GLint size;
        GLenum type;
        std::string name(256, '\0');
        glGetActiveUniform(program, i, name.size(), nullptr, &size, &type, name.data());
        CHECK_GL_ERROR();

        name.resize(strlen(name.data()));
        auto loc = glGetUniformLocation(program, name.data());
        CHECK_GL_ERROR();

        if (loc == -1) {
            continue;
        }

        UniformVar var;
        TL_TRY_ASSIGN(var, UniformVar::create(type, loc, name));
        uniforms.emplace(std::move(name), var);
    }

    return uniforms;
}
};  // namespace

Shader::Shader(GLenum type, GLuint handle, std::string_view src) noexcept
    : GLResource(handle), m_type(type), m_src(src) {
}

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
        return TL_ERROR("Shader is not valid");
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

auto Shader::from_file(ShaderType type,
                       std::string_view file) noexcept -> tl::expected<ShaderRef, std::string> {
    std::ifstream stream(file.data());
    if (!stream.is_open()) {
        return TL_ERROR("Failed to open vetex shader file");
    }

    std::string const src{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    return from_src(type, src);
}

auto Shader::clone(ViewPtr<Shader> other) noexcept -> tl::expected<ShaderRef, std::string> {
    if (!other || !other->valid()) {
        return TL_ERROR("Invalid shader");
    }
    return from_src(to_shader_type(other->m_type), other->m_src);
}

auto Shader::from_src(ShaderType type,
                      std::string_view src) noexcept -> tl::expected<ShaderRef, std::string> {
    auto const gltype = to_gl_type(type);
    GLuint handle;
    TL_TRY_ASSIGN(handle, create_shader(gltype, src));
    auto ret = ShaderRef{new Shader{gltype, handle, src}, GLResourceDeleter{}};
    return ret;
}

ShaderProgram::~ShaderProgram() noexcept {
    if (m_handle) {
        glDeleteProgram(m_handle);
    }
}

ShaderProgram::ShaderProgram(EArray<ShaderType, ShaderRef> shaders, UniformMap uniforms,
                             GLuint handle) noexcept
    : GLResource{handle}, m_shaders{std::move(shaders)}, m_uniforms{std::move(uniforms)} {
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept {
    swap(std::move(other));
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    swap(std::move(other));
    return *this;
}

auto ShaderProgram::set_uniform(std::string_view name,
                                UniformVar var) noexcept -> tl::expected<void, std::string> {
    auto const it = m_uniforms.find(name.data());
    if (it == m_uniforms.end()) {
        return TL_ERROR("Uniform not found");
    }

    TL_CHECK_AND_PASS(var.upload());
    it->second = std::move(var);
    return {};
}

auto ShaderProgram::get_uniform_map() const noexcept -> View<UniformMap> {
    return m_uniforms;
}

auto ShaderProgram::set_texture(std::string_view name, ViewPtr<GLTexture> tex,
                                GLuint slot) noexcept -> tl::expected<void, std::string> {
    if (!valid()) {
        return TL_ERROR("Invalid shader program");
    }

    glActiveTexture(GL_TEXTURE0 + slot);
    TL_CHECK_AND_PASS(tex->bind());
    TL_CHECK_AND_PASS(set_uniform(name, static_cast<int>(slot)));

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

auto ShaderProgram::try_recompile(View<ShaderDesc> new_srcs) noexcept
    -> tl::expected<void, std::string> {
    ShaderProgramRef copy;
    TL_TRY_ASSIGN(copy, ShaderProgram::from_srcs(new_srcs));
    swap(std::move(*copy));

    return {};
}

auto ShaderProgram::recompile(View<ShaderDesc> new_srcs) noexcept
    -> tl::expected<void, std::string> {
    if (!m_handle) {
        return TL_ERROR("Shader program is not valid");
    }

    for (auto const type : EIter<ShaderType>{}) {
        if (auto& src = new_srcs.get()[type]) {
            // modify or create
            if (!m_shaders[type]) {
                TL_TRY_ASSIGN(m_shaders[type], Shader::from_src(type, *src));
            } else {
                TL_CHECK_AND_PASS(m_shaders[type]->recompile(*src));
            }
        } else {
            // delete
            if (m_shaders[type]) {
                m_shaders[type] = nullptr;
            }
        }
    }
    TL_CHECK_AND_PASS(link_shaders(m_handle, m_shaders));

    // upload old uniforms
    TL_CHECK_AND_PASS(bind());
    {
        UniformMap uniforms;
        TL_TRY_ASSIGN(uniforms, get_uniforms(m_handle));
        // upload uniforms
        for (auto&& [name, var] : uniforms) {
            if (auto it = m_uniforms.find(name); it != m_uniforms.end()) {
                // transfer old value
                var.value = it->second.value;
            }
            TL_CHECK_AND_PASS(var.upload());
        }
        m_uniforms = std::move(uniforms);
    }
    unbind();

    return {};
}

auto ShaderProgram::get_stage(ShaderType type) const noexcept -> ViewPtr<Shader> {
    return m_shaders[type].get();
}

void ShaderProgram::swap(ShaderProgram&& other) noexcept {
    m_shaders.swap(other.m_shaders);
    m_uniforms.swap(other.m_uniforms);
    this->GLResource::swap(std::move(other));
}

auto ShaderProgram::from_shaders(EArray<ShaderType, ShaderRef> shaders) noexcept
    -> tl::expected<ShaderProgramRef, std::string> {
    auto const program = glCreateProgram();
    if (!program) {
        return TL_ERROR("Failed to create shader program");
    }

    TL_CHECK(link_shaders(program, shaders));

    UniformMap uniforms;
    TL_TRY_ASSIGN(uniforms, get_uniforms(program));

    return ShaderProgramRef{new ShaderProgram{std::move(shaders), std::move(uniforms), program},
                            GLResourceDeleter{}};
}

auto ShaderProgram::clone(ViewPtr<ShaderProgram> other) noexcept
    -> tl::expected<ShaderProgramRef, std::string> {
    if (!other || !other->valid()) {
        return TL_ERROR("Invalid shader program");
    }

    EArray<ShaderType, ShaderRef> shaders{};
    for (auto const type : EIter<ShaderType>{}) {
        if (other->m_shaders[type]) {
            TL_TRY_ASSIGN(shaders[type], Shader::clone(other->m_shaders[type].get()));
        }
    }
    return from_shaders(std::move(shaders));
}

auto ShaderProgram::from_files(View<ShaderDesc> files) noexcept
    -> tl::expected<ShaderProgramRef, std::string> {
    EArray<ShaderType, ShaderRef> shaders{};
    for (auto const type : EIter<ShaderType>{}) {
        if (auto& file = files.get()[type]) {
            TL_TRY_ASSIGN(shaders[type], Shader::from_file(type, *file));
        }
    }

    return from_shaders(std::move(shaders));
}

auto ShaderProgram::from_srcs(View<ShaderDesc> srcs) noexcept
    -> tl::expected<ShaderProgramRef, std::string> {
    EArray<ShaderType, ShaderRef> shaders{};
    for (auto const type : EIter<ShaderType>{}) {
        if (auto& src = srcs.get()[type]) {
            TL_TRY_ASSIGN(shaders[type], Shader::from_src(type, *src));
        }
    }

    return from_shaders(std::move(shaders));
}
