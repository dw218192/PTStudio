#pragma once
#include "glResource.h"

#include <string>
#include <tl/expected.hpp>

struct GLFrameBuffer;

using GLFrameBufferRef = GLResRef<GLFrameBuffer>;

struct GLFrameBuffer final : GLResource {
	[[nodiscard]] static auto create()->tl::expected<GLFrameBufferRef, std::string>;
	[[nodiscard]] auto bind() const noexcept->tl::expected<void, std::string>;

	GLFrameBuffer(GLFrameBuffer const&) = delete;
	auto operator=(GLFrameBuffer const&)->GLFrameBuffer & = delete;

	GLFrameBuffer(GLFrameBuffer&& other) noexcept;
	auto operator=(GLFrameBuffer&& other) noexcept -> GLFrameBuffer&;

	static void unbind() noexcept;
	[[nodiscard]] auto attach(GLenum attachment, struct GLRenderBuffer const* buf) const ->tl::expected<void, std::string>;
	[[nodiscard]] auto attach(GLenum attachment, struct GLTexture const* tex) const->tl::expected<void, std::string>;
	[[nodiscard]] auto set_draw_buffer(GLenum attachment) const ->tl::expected<void, std::string>;
private:
	GLFrameBuffer(GLuint handle) noexcept;
	~GLFrameBuffer() noexcept override;
};