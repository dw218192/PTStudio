#include "include/glBuffer.h"

auto GLBuffer::create(GLenum target) -> tl::expected<BufferRef, std::string> {
	auto ret = BufferRef{ new GLBuffer { target }, GLResourceDeleter{ } };

	GLuint buffer;
	glGenBuffers(1, &buffer);
	CHECK_GL_ERROR();

	ret->m_handle = buffer;
	return ret;
}

GLBuffer::GLBuffer(GLenum target) noexcept : GLResource{ }, m_target { target } { }

GLBuffer::~GLBuffer() noexcept {
	if(m_handle) {
		glDeleteBuffers(1, &m_handle);
	}
}