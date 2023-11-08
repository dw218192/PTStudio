#include "include/glVertexArray.h"
#include "utils.h"

using namespace PTS;

auto GLVertexArray::create(GLsizei num_vertices) -> tl::expected<GLVertexArrayRef, std::string> {
	GLuint vao;
	glGenVertexArrays(1, &vao);
	CHECK_GL_ERROR();
	return GLVertexArrayRef{ new GLVertexArray { vao, num_vertices }, GLResourceDeleter{} };
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

auto GLVertexArray::draw_array(GLenum mode) const noexcept -> tl::expected<void, std::string> {
	glDrawArrays(mode, 0, m_num_vertices);
	CHECK_GL_ERROR();
	return {};
}

void GLVertexArray::swap(GLVertexArray&& other) noexcept {
	m_num_vertices = other.m_num_vertices;
	m_buffers = std::move(other.m_buffers);
	other.m_num_vertices = 0;
	this->GLResource::swap(std::move(other));
}

GLVertexArray::GLVertexArray(GLuint handle, GLsizei num_vertices) noexcept : GLResource(handle), m_num_vertices{ num_vertices } {}

GLVertexArray::~GLVertexArray() noexcept {
	if (m_handle) {
		glDeleteVertexArrays(1, &m_handle);
	}
}
