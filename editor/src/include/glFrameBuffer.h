#pragma once
#include "glResource.h"
#include "glRenderBuffer.h"
#include "glTexture.h"

#include <string>
#include <tl/expected.hpp>
#include <unordered_map>

struct GLFrameBuffer;

using GLFrameBufferRef = GLResRef<GLFrameBuffer>;

struct AttachmentDesc {
	GLenum attachment;
	GLenum format;
};

struct GLFrameBuffer final : GLResource {
	[[nodiscard]] static auto create()->tl::expected<GLFrameBufferRef, std::string>;
	[[nodiscard]] auto bind() const noexcept->tl::expected<void, std::string>;

	GLFrameBuffer(GLFrameBuffer const&) = delete;
	auto operator=(GLFrameBuffer const&) -> GLFrameBuffer& = delete;

	GLFrameBuffer(GLFrameBuffer&& other) noexcept;
	auto operator=(GLFrameBuffer&& other) noexcept -> GLFrameBuffer&;

	static void unbind() noexcept;

	[[nodiscard]] auto attach(unsigned width, unsigned height, std::initializer_list<AttachmentDesc> attachments) ->tl::expected<void, std::string>;
	[[nodiscard]] auto set_draw_buffer(GLenum attachment) const ->tl::expected<void, std::string>;
	[[nodiscard]] auto resize(unsigned width, unsigned height) noexcept ->tl::expected<void, std::string>;

	[[nodiscard]] auto get_texture(GLenum attachment) const noexcept -> GLTexture const*;
	[[nodiscard]] auto get_render_buffer(GLenum attachment) const noexcept -> GLRenderBuffer const*;

private:
	void swap(GLFrameBuffer&& other) noexcept;
	GLFrameBuffer(GLuint handle) noexcept;
	~GLFrameBuffer() noexcept override;

	std::unordered_map<GLenum, GLRenderBufferRef> m_rbo_attchs;
	std::unordered_map<GLenum, GLTextureRef> m_tex_attchs;
};