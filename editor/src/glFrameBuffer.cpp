#include "include/glFrameBuffer.h"
#include "include/glRenderBuffer.h"
#include "include/glTexture.h"
#include "utils.h"

auto GLFrameBuffer::create() -> tl::expected<GLFrameBufferRef, std::string> {
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	CHECK_GL_ERROR();
	return GLFrameBufferRef{ new GLFrameBuffer{ fbo }, GLResourceDeleter{} };
}

auto GLFrameBuffer::bind() const noexcept -> tl::expected<void, std::string> {
	if (!m_handle) {
		return TL_ERROR("frame buffer is not valid");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
	CHECK_GL_ERROR();
	return {};
}

GLFrameBuffer::GLFrameBuffer(GLFrameBuffer&& other) noexcept {
	swap(std::move(other));
}
auto GLFrameBuffer::operator=(GLFrameBuffer&& other) noexcept -> GLFrameBuffer& {
	swap(std::move(other));
	return *this;
}

void GLFrameBuffer::unbind() noexcept {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

auto GLFrameBuffer::attach(GLenum attachment, GLRenderBuffer const* buf) const ->tl::expected<void, std::string> {
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, buf->handle());
	CHECK_GL_ERROR();
	return {};
}

auto GLFrameBuffer::attach(GLenum attachment, GLTexture const* tex) const -> tl::expected<void, std::string> {
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex->handle(), 0);
	CHECK_GL_ERROR();
	return {};
}
auto GLFrameBuffer::set_draw_buffer(GLenum attachment) const -> tl::expected<void, std::string> {
	GLenum const draw_buffer = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, &draw_buffer);
	CHECK_GL_ERROR();
	return {};
}

GLFrameBuffer::GLFrameBuffer(GLuint handle) noexcept : GLResource(handle) {}

GLFrameBuffer::~GLFrameBuffer() noexcept {
	if (m_handle) {
		glDeleteFramebuffers(1, &m_handle);
	}
}