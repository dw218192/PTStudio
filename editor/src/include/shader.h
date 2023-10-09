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
    [[nodiscard]] static auto from_shaders(Shader&& vs, Shader&& ps) noexcept -> tl::expected<ShaderProgram, std::string>;

    ShaderProgram() = default;
    ~ShaderProgram() noexcept;

    ShaderProgram(ShaderProgram&&) noexcept;
    ShaderProgram& operator=(ShaderProgram&&) noexcept;

	// shouldn't be copied because we have handles to GL resources
    ShaderProgram(ShaderProgram&) = delete;
    ShaderProgram& operator=(ShaderProgram&) = delete;

    [[nodiscard]] auto set_uniform(std::string_view name, glm::mat4 const& value) const noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto set_uniform(std::string_view name, glm::vec3 const& value) const noexcept -> tl::expected<void, std::string>;
    
    [[nodiscard]] auto valid() const noexcept { return m_handle != 0; }

    void use() const noexcept;
    void unuse() const noexcept;
private:
    ShaderProgram(Shader&& vs, Shader&& ps) noexcept : m_handle(0), m_vs{ std::move(vs) }, m_ps{ std::move(ps) } { }

    GLuint m_handle;
    Shader m_vs;
    Shader m_ps;
};