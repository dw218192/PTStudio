#pragma once

#include <GL/glew.h>
#include <string>
#include <string_view>
#include <tl/expected.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "glResource.h"
#include "enumArray.h"
#include "utils.h"
#include <tcb/span.hpp>

struct Shader;
struct ShaderProgram;

enum class ShaderType {
    Vertex,
    Fragment,
    __COUNT
};

using ShaderProgramRef = UniqueGLResRef<ShaderProgram>;
using ShaderRef = UniqueGLResRef<Shader>;

inline auto to_string(ShaderType type) noexcept {
	switch (type) {
	case ShaderType::Vertex:
        return "Vertex Shader";
	case ShaderType::Fragment:
        return "Fragment Shader";
	}
    return "";
}

struct Shader final : GLResource {
	friend struct GLResourceDeleter;
    NO_COPY(Shader);

    [[nodiscard]] static auto from_file(ShaderType type, std::string_view file) noexcept -> tl::expected<ShaderRef, std::string>;
    [[nodiscard]] static auto from_src(ShaderType type, std::string_view src) noexcept -> tl::expected<ShaderRef, std::string>;
    [[nodiscard]] static auto clone(ViewPtr<Shader> other) noexcept -> tl::expected<ShaderRef, std::string>;

    Shader(Shader&& other) noexcept;
    auto operator=(Shader&& other) noexcept -> Shader&;

    [[nodiscard]] auto recompile(std::string_view new_src) -> tl::expected<void, std::string>;
    [[nodiscard]] auto get_src() const noexcept -> std::string_view { return m_src; }
    [[nodiscard]] auto get_type() const noexcept -> ShaderType;
private:
    void swap(Shader&& other) noexcept;
    ~Shader() noexcept override;
    Shader(GLenum type, GLuint handle, std::string_view src) noexcept;

    GLenum m_type;
    std::string m_src;
};

struct ShaderProgram final : GLResource {
    template<typename T>
    using StageDesc = std::pair<ShaderType const, T> ;
    NO_COPY(ShaderProgram);


    [[nodiscard]] static auto from_srcs(tcb::span<StageDesc<std::string_view> const> srcs) noexcept
        -> tl::expected<ShaderProgramRef, std::string>;
    
    [[nodiscard]] static auto from_files(tcb::span<StageDesc<std::string_view> const> files) noexcept
        -> tl::expected<ShaderProgramRef, std::string>;
    
    [[nodiscard]] static auto from_shaders(tcb::span<StageDesc<ShaderRef>> shaders) noexcept
        -> tl::expected<ShaderProgramRef, std::string>;
    
    [[nodiscard]] static auto clone(ViewPtr<ShaderProgram> other) noexcept
		-> tl::expected<ShaderProgramRef, std::string>;

    ShaderProgram(ShaderProgram&&) noexcept;
    auto operator=(ShaderProgram&&) noexcept -> ShaderProgram&;

    template<typename UniformType>
    [[nodiscard]] auto set_uniform(std::string_view name, UniformType const& value) const noexcept
		-> tl::expected<void, std::string>;

	[[nodiscard]] auto set_texture(std::string_view name, ViewPtr<struct GLTexture> tex, GLuint slot) const noexcept
        -> tl::expected<void, std::string>;

    [[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
    static void unbind() noexcept;

    [[nodiscard]] auto try_recompile(tcb::span<StageDesc<std::string_view> const> new_srcs) noexcept
        -> tl::expected<void, std::string>;
    [[nodiscard]] auto recompile(tcb::span<StageDesc<std::string_view> const> new_srcs) noexcept
        -> tl::expected<void, std::string>;
    
    [[nodiscard]] auto get_stage(ShaderType type) const noexcept -> ViewPtr<Shader>;
private:
    void swap(ShaderProgram&& other) noexcept;
    ~ShaderProgram() noexcept override;
    ShaderProgram(EArray<ShaderType, ShaderRef> shaders, GLuint handle) noexcept;
    EArray<ShaderType, ShaderRef> m_shaders{};
};

template<typename UniformType>
auto ShaderProgram::set_uniform(std::string_view name, View<UniformType> value) const noexcept -> tl::expected<void, std::string> {
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