#pragma once

#include <GL/glew.h>

#include <any>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <tcb/span.hpp>
#include <tl/expected.hpp>
#include <unordered_map>

#include "enumArray.h"
#include "glResource.h"
#include "shaderType.h"
#include "uniformVar.h"
#include "utils.h"

namespace PTS {
using ShaderProgramRef = UniqueGLResRef<ShaderProgram>;
using ShaderRef = UniqueGLResRef<Shader>;

struct Shader final : GLResource {
    friend struct GLResourceDeleter;
    NO_COPY(Shader);

    [[nodiscard]] static auto from_file(ShaderType type, std::string_view file) noexcept
        -> tl::expected<ShaderRef, std::string>;
    [[nodiscard]] static auto from_src(ShaderType type, std::string_view src) noexcept
        -> tl::expected<ShaderRef, std::string>;
    [[nodiscard]] static auto clone(ViewPtr<Shader> other) noexcept
        -> tl::expected<ShaderRef, std::string>;

    Shader(Shader&& other) noexcept;
    auto operator=(Shader&& other) noexcept -> Shader&;

    [[nodiscard]] auto recompile(std::string_view new_src) -> tl::expected<void, std::string>;
    [[nodiscard]] auto get_src() const noexcept -> std::string_view {
        return m_src;
    }
    [[nodiscard]] auto get_type() const noexcept -> ShaderType;

   private:
    void swap(Shader&& other) noexcept;
    ~Shader() noexcept override;
    Shader(GLenum type, GLuint handle, std::string_view src) noexcept;

    GLenum m_type;
    std::string m_src;
};

struct ShaderProgram final : GLResource {
    using UniformMap = std::unordered_map<std::string, UniformVar>;
    using ShaderDesc = EArray<ShaderType, std::optional<std::string>>;

    NO_COPY(ShaderProgram);
    [[nodiscard]] static auto from_srcs(View<ShaderDesc> srcs) noexcept
        -> tl::expected<ShaderProgramRef, std::string>;

    [[nodiscard]] static auto from_files(View<ShaderDesc> files) noexcept
        -> tl::expected<ShaderProgramRef, std::string>;

    [[nodiscard]] static auto from_shaders(EArray<ShaderType, ShaderRef> shaders) noexcept
        -> tl::expected<ShaderProgramRef, std::string>;

    [[nodiscard]] static auto clone(ViewPtr<ShaderProgram> other) noexcept
        -> tl::expected<ShaderProgramRef, std::string>;

    ShaderProgram(ShaderProgram&&) noexcept;
    auto operator=(ShaderProgram&&) noexcept -> ShaderProgram&;

    /**
     * sets and uploads a uniform variable to the shader program
     */
    template <typename UniformType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<UniformType>, UniformVar>>>
    [[nodiscard]] auto set_uniform(std::string_view name, UniformType&& value) noexcept
        -> tl::expected<void, std::string>;
    [[nodiscard]] auto set_uniform(std::string_view name, UniformVar var) noexcept
        -> tl::expected<void, std::string>;

    [[nodiscard]] auto get_uniform_map() const noexcept -> View<UniformMap>;

    [[nodiscard]] auto set_texture(std::string_view name, ViewPtr<struct GLTexture> tex,
                                   GLuint slot) noexcept -> tl::expected<void, std::string>;

    [[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
    static void unbind() noexcept;

    [[nodiscard]] auto try_recompile(View<ShaderDesc> new_srcs) noexcept
        -> tl::expected<void, std::string>;

    [[nodiscard]] auto get_stage(ShaderType type) const noexcept -> ViewPtr<Shader>;

   private:
    [[nodiscard]] auto recompile(View<ShaderDesc> new_srcs) noexcept
        -> tl::expected<void, std::string>;

    void swap(ShaderProgram&& other) noexcept;

    ~ShaderProgram() noexcept override;
    ShaderProgram(EArray<ShaderType, ShaderRef> shaders, UniformMap uniforms,
                  GLuint handle) noexcept;

    EArray<ShaderType, ShaderRef> m_shaders{};
    UniformMap m_uniforms{};
};

template <typename UniformType, typename>
auto ShaderProgram::set_uniform(std::string_view name, UniformType&& value) noexcept
    -> tl::expected<void, std::string> {
    using ValueType = std::decay_t<UniformType>;

    if (!valid()) {
        return TL_ERROR("Invalid shader program");
    }

    auto const it = m_uniforms.find(name.data());
    if (it == m_uniforms.end()) {
        return TL_ERROR("Uniform not found");
    }

    it->second.set_value<ValueType>(value);
    TL_CHECK_AND_PASS(it->second.upload());
    CHECK_GL_ERROR();

    return {};
}
}  // namespace PTS
