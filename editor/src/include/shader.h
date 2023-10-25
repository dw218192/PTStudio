#pragma once

#include <GL/glew.h>
#include <string>
#include <string_view>
#include <tl/expected.hpp>
#include <tcb/span.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_map>
#include <any>

#include "glResource.h"
#include "enumArray.h"
#include "utils.h"
#include "uniformVar.h"

struct Shader;
struct ShaderProgram;
DECL_ENUM(ShaderType,
    Vertex,
    Fragment
);

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
    using UniformMap = std::unordered_map<std::string, UniformVar>;

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

    /**
     * sets and uploads a uniform variable to the shader program
    */
    template<typename UniformType, typename = std::enable_if_t<!std::is_same_v<std::decay_t<UniformType>, UniformVar>>>
    [[nodiscard]] auto set_uniform(std::string_view name, UniformType&& value) noexcept
		-> tl::expected<void, std::string>;
    [[nodiscard]] auto set_uniform(std::string_view name, UniformVar var) noexcept
        -> tl::expected<void, std::string>;

    [[nodiscard]] auto get_uniform_map() const noexcept -> View<UniformMap>;

	[[nodiscard]] auto set_texture(std::string_view name, ViewPtr<struct GLTexture> tex, GLuint slot) noexcept
        -> tl::expected<void, std::string>;

    [[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
    static void unbind() noexcept;

    [[nodiscard]] auto try_recompile(tcb::span<StageDesc<std::string_view> const> new_srcs) noexcept
        -> tl::expected<void, std::string>;

    [[nodiscard]] auto get_stage(ShaderType type) const noexcept -> ViewPtr<Shader>;

private:
	[[nodiscard]] auto recompile(tcb::span<StageDesc<std::string_view> const> new_srcs) noexcept
        -> tl::expected<void, std::string>;

    void swap(ShaderProgram&& other) noexcept;

	~ShaderProgram() noexcept override;
    ShaderProgram(EArray<ShaderType, ShaderRef> shaders, UniformMap uniforms, GLuint handle) noexcept;

	EArray<ShaderType, ShaderRef> m_shaders{};
    UniformMap m_uniforms{};
};

template<typename UniformType, typename>
auto ShaderProgram::set_uniform(std::string_view name, UniformType&& value) noexcept -> tl::expected<void, std::string> {
    using ValueType = std::decay_t<UniformType>;

	if (!valid()) {
        return TL_ERROR( "Invalid shader program" );
    }

    auto const it = m_uniforms.find(name.data());
    if (it == m_uniforms.end()) {
        return TL_ERROR( "Uniform not found" );
    }
    GLint loc = it->second.get_loc(); 
    if (loc == -1) {
        return TL_ERROR( "Invalid uniform location" );
    }
    if ((type_to_enum_msk(value) & it->second.get_type()) == 0) {
        return TL_ERROR( "Uniform type mismatch" );
    }

    if constexpr (std::is_same_v<ValueType, glm::mat3>) {
        glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<ValueType, glm::mat4>) {
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<ValueType, glm::vec2>) {
        glUniform2fv(loc, 1, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<ValueType, glm::vec3>) {
        glUniform3fv(loc, 1, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<ValueType, glm::vec4>) {
        glUniform4fv(loc, 1, glm::value_ptr(value));
    }
    else if constexpr (std::is_same_v<ValueType, float>) {
        glUniform1f(loc, value);
    }
    else if constexpr (std::is_same_v<ValueType, int>) {
        glUniform1i(loc, value);
    }
    else {
        static_assert(false, "Unsupported uniform type");
    }
    CHECK_GL_ERROR();

    it->second.set_value<ValueType>(value);
    return {};
}