#pragma once

#include <tcb/span.hpp>
#include <tl/expected.hpp>
#include <string>

#include "glResource.h"
#include "utils.h"

struct GLBuffer;
using GLBufferRef = GLResRef<GLBuffer>;

struct GLBuffer final : GLResource {
	static auto create(GLenum target)-> tl::expected<GLBufferRef, std::string>;

	GLBuffer(GLBuffer const&) = delete;
	auto operator=(GLBuffer const&) ->GLBuffer& = delete;
	GLBuffer(GLBuffer&& other) noexcept;
	auto operator=(GLBuffer&& other) noexcept -> GLBuffer&;

	template<typename T, size_t Extent>
	[[nodiscard]] auto set_data(tcb::span<T const, Extent> data, GLenum usage = GL_STATIC_DRAW) -> tl::expected<void, std::string>;

	[[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
	void unbind() const noexcept;

	[[nodiscard]] auto size() const noexcept { return m_size;  }
private:
	void swap(GLBuffer&& other) noexcept;
	GLBuffer(GLenum target, GLuint handle) noexcept;
	~GLBuffer() noexcept override;

	GLuint m_target;
	size_t m_size{ 0 };
};

template <typename T, size_t Extent>
auto GLBuffer::set_data(tcb::span<T const, Extent> data, GLenum usage) -> tl::expected<void, std::string> {
	if (data.empty()) {
		return {};
	}
	glBufferData(m_target, data.size() * sizeof(T), data.data(), usage);
	CHECK_GL_ERROR();

	m_size = data.size();
	return {};
}
