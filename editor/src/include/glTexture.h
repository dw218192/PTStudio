#pragma once

#include "glResource.h"
#include "texture.h"

struct GLTexture;

using GLTextureRef = UniqueGLResRef<GLTexture>;

struct GLTexture final : Texture, GLResource {
    [[nodiscard]] static auto create(unsigned width, unsigned height, GLenum format, std::initializer_list<GLParam> params = {
        { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
        { GL_TEXTURE_MAG_FILTER, GL_LINEAR }
    }) noexcept-> tl::expected<GLTextureRef, std::string>;

    GLTexture(GLTexture const&) = delete;
    auto operator=(GLTexture const&) -> GLTexture& = delete;

    GLTexture(GLTexture&& other) noexcept;
    auto operator=(GLTexture&& other) noexcept -> GLTexture&;

    [[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string> override;
    void unbind() const noexcept override;
    [[nodiscard]] auto get_id() const noexcept -> void* override;
    [[nodiscard]] auto format() const noexcept -> GLenum { return m_format; }
    [[nodiscard]] auto resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> override;

private:
    [[nodiscard]] static auto create_tex(unsigned width, unsigned height, GLenum format, std::initializer_list<GLParam> params) noexcept
		-> tl::expected<GLuint, std::string>;
    void swap(GLTexture&& other) noexcept;

    GLTexture(unsigned width, unsigned height, GLenum format, GLuint handle);
    ~GLTexture() noexcept override;
    [[nodiscard]] auto fetch_pixels() const noexcept -> tl::expected<void, std::string> override;

    GLenum m_format;
};