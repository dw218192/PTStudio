#include "include/glVertexArray.h"
#include "utils.h"

auto GLVertexArray::create() -> tl::expected<GLVertexArrayRef, std::string> {
	GLuint vao;
	glGenVertexArrays(1, &vao);
	CHECK_GL_ERROR();
	return GLVertexArrayRef{ new GLVertexArray { vao }, GLResourceDeleter{} };
}

GLVertexArray::GLVertexArray(GLVertexArray&& other) noexcept {
	swap(std::move(other));
}

auto GLVertexArray::operator=(GLVertexArray&& other) noexcept -> GLVertexArray& {
	swap(std::move(other));
	return *this;
}

auto GLVertexArray::bind() const noexcept -> tl::expected<void, std::string> {
	if (!m_handle) {
		return TL_ERROR("vertex array is not valid");
	}
	glBindVertexArray(m_handle);
	CHECK_GL_ERROR();
	return {};
}

void GLVertexArray::unbind() noexcept {
	glBindVertexArray(0);
}

void GLVertexArray::swap(GLVertexArray&& other) noexcept {
	this->GLResource::swap(std::move(other));
}

GLVertexArray::GLVertexArray(GLuint handle) noexcept : GLResource(handle) {}

GLVertexArray::~GLVertexArray() noexcept {
	if (m_handle) {
		glDeleteVertexArrays(1, &m_handle);
	}
}
