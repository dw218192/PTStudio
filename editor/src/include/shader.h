#pragma once

#include <GL/glew.h>
#include <string>
#include <string_view>
#include <tl/expected.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "glResource.h"
#include "utils.h"

struct Shader;
struct ShaderProgram;

enum class ShaderType {
    Vertex,
    Fragment
};

using ShaderProgramRef = UniqueGLResRef<ShaderProgram>;
using ShaderRef = UniqueGLResRef<Shader>;

struct Shader final : GLResource {
	friend struct GLResourceDeleter;

    [[nodiscard]] static auto from_file(ShaderType type, std::string_view file) noexcept -> tl::expected<ShaderRef, std::string>;
    [[nodiscard]] static auto from_src(ShaderType type, std::string_view src) noexcept -> tl::expected<ShaderRef, std::string>;
    [[nodiscard]] static auto clone(Shader const* other) noexcept -> tl::expected<ShaderRef, std::string>;

    Shader(Shader const&) = delete;
    auto operator=(Shader const&) ->Shader& = delete;

    Shader(Shader&& other) noexcept;
    auto operator=(Shader&& other) noexcept -> Shader&;

    [[nodiscard]] auto recompile(std::string_view new_src) -> tl::expected<void, std::string>;
private:
    void swap(Shader&& other) noexcept;
    ~Shader() noexcept override;
    Shader(GLenum type, GLuint handle, std::string_view src) noexcept;

    GLenum m_type;
    std::string m_src;
};

struct ShaderProgram final : GLResource {
    [[nodiscard]] static auto from_srcs(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgramRef, std::string>;
    [[nodiscard]] static auto from_files(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgramRef, std::string>;
    [[nodiscard]] static auto from_shaders(ShaderRef vs, ShaderRef ps) noexcept -> tl::expected<ShaderProgramRef, std::string>;
    [[nodiscard]] static auto clone(ShaderProgram const* other) noexcept -> tl::expected<ShaderProgramRef, std::string>;

    ShaderProgram(ShaderProgram const&) = delete;
    auto operator=(ShaderProgram const&)->ShaderProgram& = delete;

    ShaderProgram(ShaderProgram&&) noexcept;
    auto operator=(ShaderProgram&&) noexcept -> ShaderProgram&;

    template<typename UniformType>
    [[nodiscard]] auto set_uniform(std::string_view name, UniformType const& value) const noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto set_texture(std::string_view name, struct GLTexture const* tex, GLuint slot) const noexcept -> tl::expected<void, std::string>;

    [[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
    static void unbind() noexcept;

    [[nodiscard]] auto recompile(std::string_view new_vs, std::string_view new_ps) noexcept -> tl::expected<void, std::string>;
private:
    void swap(ShaderProgram&& other) noexcept;
    ~ShaderProgram() noexcept override;
    ShaderProgram(ShaderRef vs, ShaderRef ps, GLuint handle) noexcept;
    ShaderRef m_vs;
    ShaderRef m_ps;
};

template<typename UniformType>
auto ShaderProgram::set_uniform(std::string_view name, UniformType const& value) const noexcept -> tl::expected<void, std::string> {
	if (!valid()) {
        return TL_ERROR( "Invalid shader program" );
    }
    GLint loc = glGetUniformLocation(m_handle, name.data());
    if (loc == -1) {
        return TL_ERROR( "Failed to get uniform location" );
    }
    if constexpr (std::is_same_v<UniformType, glm::mat4>) {
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<UniformType, glm::vec2>) {
        glUniform2fv(loc, 1, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<UniformType, glm::vec3>) {
        glUniform3fv(loc, 1, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<UniformType, glm::vec4>) {
        glUniform4fv(loc, 1, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<UniformType, float>) {
        glUniform1f(loc, value);
    }
    else if constexpr (std::is_same_v<UniformType, int>) {
        glUniform1i(loc, value);
    }
    else {
        static_assert(false, "Unsupported uniform type");
    }

    CHECK_GL_ERROR();

    return {};
}