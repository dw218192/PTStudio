#pragma once
#include "glResource.h"
#include "glRenderBuffer.h"
#include "glTexture.h"

#include <string>
#include <tl/expected.hpp>
#include <unordered_map>
#include <glm/vec3.hpp>

struct GLFrameBuffer;

using GLFrameBufferRef = GLResRef<GLFrameBuffer>;

struct AttachmentDesc {
	GLenum attachment;
	GLenum format;
};

/**
 * @brief the default frame buffer object that is created by the window
 * @details rendering to this buffer is the same as rendering to the screen
 */
namespace MainFrameBuffer {
	[[nodiscard]] auto bind() noexcept -> tl::expected<void, std::string>;
	
	/**
	 * @brief clear the frame buffer
	 * @param color the color to clear to
	 * @param depth the depth to clear to
	 * @return error message on failure
	 */
	[[nodiscard]] auto clear(glm::vec3 color, float depth) noexcept -> tl::expected<void, std::string>;
	void set(GLbitfield mask);

	static inline GLbitfield s_attachment_flags{ 0 };
	static inline GLuint s_prev_fbo{ 0 };
}

/**
 * @brief a frame buffer object
 * @details a frame buffer object is a collection of buffers that can be used as the destination for rendering\N
 * only supports color render buffers and depth render buffers for now
*/
struct GLFrameBuffer final : GLResource {
	[[nodiscard]] static auto create()->tl::expected<GLFrameBufferRef, std::string>;
	[[nodiscard]] auto bind() const noexcept->tl::expected<void, std::string>;

	GLFrameBuffer(GLFrameBuffer const&) = delete;
	auto operator=(GLFrameBuffer const&) -> GLFrameBuffer& = delete;

	GLFrameBuffer(GLFrameBuffer&& other) noexcept;
	auto operator=(GLFrameBuffer&& other) noexcept -> GLFrameBuffer&;

	static void unbind() noexcept;

	[[nodiscard]] auto attach(unsigned width, unsigned height, std::initializer_list<AttachmentDesc> descs) ->tl::expected<void, std::string>;
	[[nodiscard]] auto set_draw_buffer(GLenum attachment) const ->tl::expected<void, std::string>;
	
	/**
	 * @brief resize the frame buffer, including any attachment if necessary
	 * @param width the new width
	 * @param height the new height
	 * @return error message on failure
	 */
	[[nodiscard]] auto resize(unsigned width, unsigned height) noexcept ->tl::expected<void, std::string>;
	[[nodiscard]] auto get_texture(GLenum attachment) const noexcept -> GLTexture const*;
	[[nodiscard]] auto get_render_buffer(GLenum attachment) const noexcept -> GLRenderBuffer const*;

	/**
	 * @brief swap the render buffer attached to the given attachment point
	 * @param attachment the attachment point
	 * @param buf the new render buffer; if nullptr, the attachment point will be cleared
	 * @return the old render buffer
	 */
	auto swap_render_buffer(GLenum attachment, GLRenderBufferRef buf) noexcept -> tl::expected<GLRenderBufferRef, std::string>;

	/**
	 * @brief swap the texture attached to the given attachment point
	 * @param attachment the attachment point
	 * @param tex the new texture; if nullptr, the attachment point will be cleared
	 * @return the old texture
	 */
	auto swap_texture(GLenum attachment, GLTextureRef tex) noexcept -> tl::expected<GLTextureRef, std::string>;

	/**
	 * @brief clear the frame buffer
	 * @param color the color to clear to
	 * @param depth the depth to clear to
	 * @return error message on failure
	 */
	auto clear(glm::vec3 color, float depth) const noexcept -> tl::expected<void, std::string>;

private:
	void swap(GLFrameBuffer&& other) noexcept;
	GLFrameBuffer(GLuint handle) noexcept;
	~GLFrameBuffer() noexcept override;

	std::unordered_map<GLenum, GLRenderBufferRef> m_rbo_attchs;
	std::unordered_map<GLenum, GLTextureRef> m_tex_attchs;

	GLbitfield m_attachment_flags{ 0 };
};