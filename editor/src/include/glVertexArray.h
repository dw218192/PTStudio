#pragma once

#include <string>
#include <tcb/span.hpp>
#include <tl/expected.hpp>
#include <glm/glm.hpp>

#include "glBuffer.h"
#include "glResource.h"

struct GLVertexArray;
using GLVertexArrayRef = GLResRef<GLVertexArray>;

struct GLVertexArray final : GLResource {
	static auto create() -> tl::expected<GLVertexArrayRef, std::string>;

	GLVertexArray(GLVertexArray const&) = delete;
	auto operator=(GLVertexArray const&) -> GLVertexArray& = delete;
	GLVertexArray(GLVertexArray&& other) noexcept;
	auto operator=(GLVertexArray&& other) noexcept -> GLVertexArray&;

	[[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
	static void unbind() noexcept;

	template<typename T>
	[[nodiscard]] auto fill(GLuint slot, tcb::span<T const> data, GLuint stride = 0) noexcept -> tl::expected<void, std::string>;

private:
	void swap(GLVertexArray&& other) noexcept;
	GLVertexArray(GLuint handle) noexcept;
	~GLVertexArray() noexcept override;

	std::unordered_map<GLuint, GLBufferRef> m_slots;
};

template <typename T>
struct GLAttributeInfo;

template <>
struct GLAttributeInfo<float> {
	static constexpr GLuint components = 1;
	static constexpr GLenum type = GL_FLOAT;
};

template <>
struct GLAttributeInfo<glm::vec2> {
	static constexpr GLuint components = 2;
	static constexpr GLenum type = GL_FLOAT;
};

template <>
struct GLAttributeInfo<glm::vec3> {
	static constexpr GLuint components = 3;
	static constexpr GLenum type = GL_FLOAT;
};

template <typename T>
auto GLVertexArray::fill(GLuint slot, tcb::span<T const> data, GLuint stride) noexcept -> tl::expected<void, std::string> {
	GLBufferRef buf;
	TL_CHECK_RET(GLBuffer::create(GL_ARRAY_BUFFER), buf);
	TL_CHECK(buf->bind());
	{
		buf->set_data(data);

		glEnableVertexAttribArray(slot);
		glVertexAttribPointer(slot, GLAttributeInfo<T>::components, GLAttributeInfo<T>::type, GL_FALSE, 0, nullptr);

		CHECK_GL_ERROR();
	}
	buf->unbind();
	m_slots[slot] = std::move(buf);
	return {};
}
