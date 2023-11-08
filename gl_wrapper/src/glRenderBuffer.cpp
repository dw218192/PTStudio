#include "glRenderBuffer.h"
#include "utils.h"

using namespace PTS;

auto GLRenderBuffer::create(unsigned width, unsigned height, GLenum format) -> tl::expected<GLRenderBufferRef, std::string> {
	GLuint rbo;
	TL_TRY_ASSIGN(rbo, create_buf(width, height, format));
	auto ret = GLRenderBufferRef{ new GLRenderBuffer{width, height, format, rbo}, GLResourceDeleter{} };
	return ret;
}

GLRenderBuffer::GLRenderBuffer(GLRenderBuffer&& other) noexcept {
	swap(std::move(other));
}
auto GLRenderBuffer::operator=(GLRenderBuffer&& other) noexcept -> GLRenderBuffer& {
	swap(std::move(other));
	return *this;
}
auto GLRenderBuffer::resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> {
	m_width = width;
	m_height = height;
	glRenderbufferStorage(GL_RENDERBUFFER, m_format, width, height);
	CHECK_GL_ERROR();

	return {};
}

auto GLRenderBuffer::bind() const noexcept ->tl::expected<void, std::string> {
	if (!m_handle) {
		return TL_ERROR("render buffer is not valid");
	}

	glBindRenderbuffer(GL_RENDERBUFFER, m_handle);
	CHECK_GL_ERROR();
	return {};
}
void GLRenderBuffer::unbind() noexcept {
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

auto GLRenderBuffer::create_buf(unsigned width, unsigned height, GLenum format) noexcept -> tl::expected<GLuint, std::string> {
	GLuint rbo;
	// create both depth and stencil buffers
	glGenRenderbuffers(1, &rbo);
	CHECK_GL_ERROR();

	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	{
		glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
		CHECK_GL_ERROR();
	}
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	return rbo;
}

void GLRenderBuffer::swap(GLRenderBuffer&& other) noexcept {
	if (this == &other) {
		return;
	}
	std::swap(m_width, other.m_width);
	std::swap(m_height, other.m_height);
	std::swap(m_format, other.m_format);
	this->GLResource::swap(std::move(other));
}

GLRenderBuffer::GLRenderBuffer(unsigned width, unsigned height, GLenum format, GLuint handle) noexcept
	: GLResource(handle), m_width(width), m_height(height), m_format(format) {}

GLRenderBuffer::~GLRenderBuffer() noexcept {
	if (m_handle) {
		glDeleteRenderbuffers(1, &m_handle);
	}
}
