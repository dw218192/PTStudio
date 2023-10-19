#include "include/glBuffer.h"

auto GLBuffer::create(GLenum target) -> tl::expected<GLBufferRef, std::string> {
	GLuint buffer;
	glGenBuffers(1, &buffer);
	CHECK_GL_ERROR();

	return GLBufferRef{ new GLBuffer { target, buffer }, GLResourceDeleter{ } };
}

GLBuffer::GLBuffer(GLBuffer&& other) noexcept {
	swap(std::move(other));
}

auto GLBuffer::operator=(GLBuffer&& other) noexcept -> GLBuffer& {
	swap(std::move(other));
	return *this;
}

auto GLBuffer::bind() const noexcept -> tl::expected<void, std::string> {
	if (!m_handle) {
		return TL_ERROR("GLBuffer is not valid");
	}
	glBindBuffer(m_target, m_handle);
	CHECK_GL_ERROR();
	return {};
}

void GLBuffer::unbind() const noexcept {
	glBindBuffer(m_target, 0);
}

void GLBuffer::swap(GLBuffer&& other) noexcept {
	m_target = other.m_target;
	m_handle = other.m_handle;
	other.m_target = 0;
	other.m_handle = 0;
	this->GLResource::swap(std::move(other));
}
GLBuffer::GLBuffer(GLenum target, GLuint handle) noexcept : GLResource{ handle }, m_target { target } { }

GLBuffer::~GLBuffer() noexcept {
	if(m_handle) {
		glDeleteBuffers(1, &m_handle);
	}
}