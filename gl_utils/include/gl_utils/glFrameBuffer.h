#pragma once
#include <glm/vec3.hpp>
#include <string>
#include <tl/expected.hpp>
#include <unordered_map>

#include "glRenderBuffer.h"
#include "glResource.h"
#include "glTexture.h"

namespace PTS {
struct GLFrameBuffer;

using GLFrameBufferRef = UniqueGLResRef<GLFrameBuffer>;

struct AttachmentDesc {
    GLenum attachment;
    GLenum format;
    std::initializer_list<GLParam> params;
};

/**
 * @brief a frame buffer object
 * @details a frame buffer object is a collection of buffers that can be used as the destination for
 * rendering\N only supports color render buffers and depth render buffers for now
 */
struct GLFrameBuffer final : GLResource {
    NODISCARD static auto create() -> tl::expected<GLFrameBufferRef, std::string>;
    NODISCARD auto bind() const noexcept -> tl::expected<void, std::string>;

    GLFrameBuffer(GLFrameBuffer const&) = delete;
    auto operator=(GLFrameBuffer const&) -> GLFrameBuffer& = delete;

    GLFrameBuffer(GLFrameBuffer&& other) noexcept;
    auto operator=(GLFrameBuffer&& other) noexcept -> GLFrameBuffer&;

    static void unbind() noexcept;

    NODISCARD auto attach(unsigned width, unsigned height,
                          std::initializer_list<AttachmentDesc> descs)
        -> tl::expected<void, std::string>;
    NODISCARD auto set_draw_buffer(GLenum attachment) const -> tl::expected<void, std::string>;

    /**
     * @brief resize the frame buffer, including any attachment if necessary
     * @param width the new width
     * @param height the new height
     * @return error message on failure
     */
    NODISCARD auto resize(unsigned width, unsigned height) noexcept
        -> tl::expected<void, std::string>;
    NODISCARD auto get_texture(GLenum attachment) const noexcept -> GLTexture const*;
    NODISCARD auto get_render_buffer(GLenum attachment) const noexcept -> GLRenderBuffer const*;

    /**
     * @brief swap the render buffer attached to the given attachment point
     * @param attachment the attachment point
     * @param buf the new render buffer; if nullptr, the attachment point will be cleared
     * @return the old render buffer
     */
    auto swap_render_buffer(GLenum attachment, GLRenderBufferRef buf) noexcept
        -> tl::expected<GLRenderBufferRef, std::string>;

    /**
     * @brief swap the texture attached to the given attachment point with the given texture
     * @param attachment the attachment point
     * @param tex the new texture; if nullptr, the attachment point will be cleared
     * @return the old texture
     */
    auto swap_texture(GLenum attachment, GLTextureRef tex) noexcept
        -> tl::expected<GLTextureRef, std::string>;

    /**
     * @brief swap the textures attached to the given attachment points
     * @param attachment1 the first attachment point
     * @param attachment2 the second attachment point
     * @return error message on failure
     */
    auto swap_texture(GLenum attachment1, GLenum attachment2) noexcept
        -> tl::expected<void, std::string>;

    /**
     * @brief clear the frame buffer, including all attachments
     * @param color the color to clear to
     * @param depth the depth to clear to
     * @return error message on failure
     */
    auto clear(glm::vec3 color, float depth) const noexcept -> tl::expected<void, std::string>;
    /**
     * @brief clear the given color attachment
     * @param color the color to clear to
     * @param depth the depth to clear to
     * @return error message on failure
     */
    auto clear_color(GLenum attachment, glm::vec3 color) const noexcept
        -> tl::expected<void, std::string>;

   private:
    void swap(GLFrameBuffer&& other) noexcept;
    GLFrameBuffer(GLuint handle) noexcept;
    ~GLFrameBuffer() noexcept override;

    std::unordered_map<GLenum, GLRenderBufferRef> m_rbo_attchs;
    std::unordered_map<GLenum, GLTextureRef> m_tex_attchs;
};
}  // namespace PTS
