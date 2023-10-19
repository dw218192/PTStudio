#pragma once

#include "glResource.h"
#include "texture.h"

struct GLTexture;
using GLTextureRef = GLResRef<GLTexture>;

struct GLTexture final : Texture, GLResource {
    [[nodiscard]] static auto create(unsigned width, unsigned height) -> tl::expected<GLTextureRef, std::string>;

    GLTexture(unsigned width, unsigned height);

    GLTexture(GLTexture const&) = delete;
    GLTexture& operator=(GLTexture const&) = delete;

    GLTexture(GLTexture&& other) noexcept;
    GLTexture& operator=(GLTexture&& other) noexcept;

    void bind() const noexcept override;
    void unbind() const noexcept override;
    auto get_handle() const noexcept -> void* override;
    [[nodiscard]] auto resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> override;

protected:
    auto static create_tex(unsigned width, unsigned height) noexcept -> tl::expected<GLuint, std::string>;
    void swap(GLTexture&& other) noexcept;
    ~GLTexture() noexcept override;
    [[nodiscard]] auto fetch_pixels() const noexcept -> tl::expected<void, std::string> override;
};