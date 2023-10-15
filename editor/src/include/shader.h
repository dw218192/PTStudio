#pragma once

#include <GL/glew.h>
#include <string>
#include <string_view>
#include <tl/expected.hpp>

#include "ext.h"

enum class ShaderType {
    Vertex,
    Fragment
};

using ShaderProgramRef = std::reference_wrapper<struct ShaderProgram>;

struct Shader {
    friend struct ShaderProgram;

    [[nodiscard]] static auto from_file(ShaderType type, std::string_view file) noexcept -> tl::expected<Shader, std::string>;
    [[nodiscard]] static auto from_src(ShaderType type, std::string_view src) noexcept -> tl::expected<Shader, std::string>;

    Shader() = default;
	~Shader() noexcept;

    Shader(Shader&&) noexcept;
    Shader& operator=(Shader&&) noexcept;

    // shouldn't be copied because we have handles to GL resources
    Shader(Shader&) = delete;
    Shader& operator=(Shader&) = delete;


    [[nodiscard]] auto valid() const noexcept -> bool { return m_handle != 0; }

private:
    Shader(ShaderType type) noexcept;
    GLenum m_type;
    GLuint m_handle;
};

struct ShaderProgram {
    [[nodiscard]] static auto from_srcs(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgram, std::string>;
    [[nodiscard]] static auto from_files(std::string_view vs, std::string_view ps) noexcept -> tl::expected<ShaderProgram, std::string>;
    [[nodiscard]] static auto from_shaders(Shader&& vs, Shader&& ps) noexcept -> tl::expected<ShaderProgram, std::string>;

    ShaderProgram() = default;
    ~ShaderProgram() noexcept;

    ShaderProgram(ShaderProgram&&) noexcept;
    ShaderProgram& operator=(ShaderProgram&&) noexcept;

	// shouldn't be copied because we have handles to GL resources
    ShaderProgram(ShaderProgram&) = delete;
    ShaderProgram& operator=(ShaderProgram&) = delete;

    template<typename UniformType>
    [[nodiscard]] auto set_uniform(std::string_view name, UniformType const& value) const noexcept -> tl::expected<void, std::string>;
    
    [[nodiscard]] auto valid() const noexcept { return m_handle != 0; }

    void use() const noexcept;
    void unuse() const noexcept;
private:
    ShaderProgram(Shader&& vs, Shader&& ps) noexcept : m_handle(0), m_vs{ std::move(vs) }, m_ps{ std::move(ps) } { }

    GLuint m_handle;
    Shader m_vs;
    Shader m_ps;
};

template<typename UniformType>
auto ShaderProgram::set_uniform(std::string_view name, UniformType const& value) const noexcept -> tl::expected<void, std::string> {
    if (!valid()) {
        return tl::unexpected{ "Invalid shader program" };
    }
    GLint loc = glGetUniformLocation(m_handle, name.data());
    if (loc == -1) {
        return tl::unexpected{ "Failed to get uniform location" };
    }
    if constexpr (std::is_same_v<UniformType, glm::mat4>) {
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(value));
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