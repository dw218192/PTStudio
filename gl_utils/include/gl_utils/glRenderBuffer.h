#pragma once
#include <core/legacy/utils.h>

#include <string>

#include "glResource.h"

namespace PTS {
struct GLRenderBuffer;
using GLRenderBufferRef = UniqueGLResRef<GLRenderBuffer>;

struct GLRenderBuffer final : GLResource {
    NODISCARD static auto create(unsigned width, unsigned height,
                                 GLenum format) -> tl::expected<GLRenderBufferRef, std::string>;

    GLRenderBuffer(GLRenderBuffer const&) = delete;
    auto operator=(GLRenderBuffer const&) -> GLRenderBuffer& = delete;

    GLRenderBuffer(GLRenderBuffer&& other) noexcept;
    auto operator=(GLRenderBuffer&& other) noexcept -> GLRenderBuffer&;

    NODISCARD auto resize(unsigned width,
                          unsigned height) noexcept -> tl::expected<void, std::string>;
    NODISCARD auto bind() const noexcept -> tl::expected<void, std::string>;
    static void unbind() noexcept;

    NODISCARD auto get_format() const noexcept -> GLenum {
        return m_format;
    }
    NODISCARD auto get_width() const noexcept -> unsigned {
        return m_width;
    }
    NODISCARD auto get_height() const noexcept -> unsigned {
        return m_height;
    }

   private:
    NODISCARD static auto create_buf(unsigned width, unsigned height,
                                     GLenum format) noexcept -> tl::expected<GLuint, std::string>;
    void swap(GLRenderBuffer&& other) noexcept;
    GLRenderBuffer(unsigned width, unsigned height, GLenum format, GLuint handle) noexcept;
    ~GLRenderBuffer() noexcept override;

    unsigned m_width, m_height;
    GLenum m_format;
};
}  // namespace PTS
