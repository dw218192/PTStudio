#pragma once

#include <core/legacy/texture.h>
#include <stb_image.h>

#include <tcb/span.hpp>
#include <tl/expected.hpp>
#include <type_traits>

#include "glResource.h"

namespace PTS {
struct GLTexture;
using GLTextureRef = UniqueGLResRef<GLTexture>;

struct GLTexture final : Texture, GLResource {
    static constexpr std::initializer_list<GLParam> k_default_params = {
        {GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}};

    NODISCARD static auto create(unsigned width, unsigned height, GLenum format,
                                 std::initializer_list<GLParam> params = k_default_params) noexcept
        -> tl::expected<GLTextureRef, std::string>;

    template <typename Byte, size_t Extent,
              typename = std::enable_if_t<std::is_integral_v<Byte> && sizeof(Byte) == 1>>
    NODISCARD static auto create(tcb::span<Byte, Extent> img_mem, FileFormat type,
                                 std::initializer_list<GLParam> params = k_default_params) noexcept
        -> tl::expected<GLTextureRef, std::string>;
    NODISCARD static auto create(std::string_view img_file, FileFormat type,
                                 std::initializer_list<GLParam> params = k_default_params) noexcept
        -> tl::expected<GLTextureRef, std::string>;
    NODISCARD static auto create(unsigned width, unsigned height, GLenum format,
                                 GLuint handle) noexcept -> tl::expected<GLTextureRef, std::string>;

    GLTexture(GLTexture const&) = delete;
    auto operator=(GLTexture const&) -> GLTexture& = delete;

    GLTexture(GLTexture&& other) noexcept;
    auto operator=(GLTexture&& other) noexcept -> GLTexture&;

    NODISCARD auto bind() const noexcept -> tl::expected<void, std::string> override;
    void unbind() const noexcept override;
    NODISCARD auto get_id() const noexcept -> void* override;
    NODISCARD auto format() const noexcept -> GLenum {
        return m_format;
    }
    NODISCARD auto resize(unsigned width,
                          unsigned height) noexcept -> tl::expected<void, std::string> override;

   private:
    NODISCARD static auto create_tex(
        unsigned width, unsigned height, GLenum format, unsigned char const* data,
        std::initializer_list<GLParam> params) noexcept -> tl::expected<GLuint, std::string>;

    void swap(GLTexture&& other) noexcept;
    GLTexture(unsigned width, unsigned height, unsigned channels, GLenum format, GLuint handle);
    ~GLTexture() noexcept override;
    NODISCARD auto fetch_pixels() const noexcept -> tl::expected<void, std::string> override;

    GLenum m_format;
};

template <typename Byte, size_t Extent, typename>
auto GLTexture::create(tcb::span<Byte, Extent> img_mem, FileFormat type,
                       std::initializer_list<GLParam> params) noexcept
    -> tl::expected<GLTextureRef, std::string> {
    unsigned width, height, num_channels;
    auto const data = stbi_load_from_memory(
        reinterpret_cast<stbi_uc const*>(img_mem.data()), static_cast<int>(img_mem.size()),
        reinterpret_cast<int*>(&width), reinterpret_cast<int*>(&height),
        reinterpret_cast<int*>(&num_channels), 0);
    if (!data) {
        return TL_ERROR("failed to load image");
    }

    auto w = width, h = height;
    // flip pixels vertically, because OpenGL expects the first pixel to be at the bottom left
    // corner
    for (auto y = 0u; y < h / 2; ++y) {
        for (auto x = 0u; x < w; ++x) {
            auto const frm = y * w + x;
            auto const to = (h - y - 1) * w + x;
            for (auto k = 0u; k < num_channels; ++k) {
                std::swap(data[num_channels * frm + k], data[num_channels * to + k]);
            }
        }
    }

    GLenum format;
    switch (num_channels) {
        case 1:
            format = GL_RED;
            break;
        case 2:
            format = GL_RG;
            break;
        case 3:
            format = GL_RGB;
            break;
        case 4:
            format = GL_RGBA;
            break;
        default:
            return TL_ERROR("unsupported image format");
    }
    GLuint tex;
    TL_TRY_ASSIGN(tex, create_tex(width, height, format, data, params));

    stbi_image_free(data);
    auto ret =
        GLTextureRef{new GLTexture{width, height, num_channels, format, tex}, GLResourceDeleter{}};
    return ret;
}
}  // namespace PTS
