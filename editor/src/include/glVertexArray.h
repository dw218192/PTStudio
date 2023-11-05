#pragma once

#include <string>
#include <tcb/span.hpp>
#include <tl/expected.hpp>
#include <glm/glm.hpp>

#include "glBuffer.h"
#include "glResource.h"

struct GLVertexArray;
using GLVertexArrayRef = UniqueGLResRef<GLVertexArray>;

struct GLVertexArray final : GLResource {
	[[nodiscard]] static auto create(GLsizei num_vertices) -> tl::expected<GLVertexArrayRef, std::string>;

	GLVertexArray(GLVertexArray const&) = delete;
	auto operator=(GLVertexArray const&) -> GLVertexArray& = delete;
	GLVertexArray(GLVertexArray&& other) noexcept;
	auto operator=(GLVertexArray&& other) noexcept -> GLVertexArray&;

	[[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
	static void unbind() noexcept;

	template <typename VertexType, std::size_t Extent, typename... GLAttributeInfos>
	auto connect(tcb::span<VertexType const, Extent> raw_data, GLAttributeInfos... attris) noexcept -> tl::expected<void, std::string>;

	auto draw_array(GLenum mode) const noexcept -> tl::expected<void, std::string>;

private:
	void swap(GLVertexArray&& other) noexcept;
	GLVertexArray(GLuint handle, GLsizei num_vertices) noexcept;
	~GLVertexArray() noexcept override;

	GLsizei m_num_vertices;
	std::vector<GLBufferRef> m_buffers;
};

struct GLAttributeInfoBase {
	GLuint slot, stride, offset;
	GLAttributeInfoBase(GLuint slot, GLuint stride, GLuint offset) : slot(slot), stride(stride), offset(offset) {}
};

template <typename T>
struct GLAttributeInfo {};

template <>
struct GLAttributeInfo<float> : GLAttributeInfoBase {
	using GLAttributeInfoBase::GLAttributeInfoBase;

	static constexpr GLuint components = 1;
	static constexpr GLenum type = GL_FLOAT;
};

template <>
struct GLAttributeInfo<glm::vec2> : GLAttributeInfoBase {
	using GLAttributeInfoBase::GLAttributeInfoBase;

	static constexpr GLuint components = 2;
	static constexpr GLenum type = GL_FLOAT;
};

template <>
struct GLAttributeInfo<glm::vec3> : GLAttributeInfoBase {
	using GLAttributeInfoBase::GLAttributeInfoBase;

	static constexpr GLuint components = 3;
	static constexpr GLenum type = GL_FLOAT;
};

template <typename VertexType, std::size_t Extent, typename... GLAttributeInfos>
auto GLVertexArray::connect(tcb::span<VertexType const, Extent> raw_data, GLAttributeInfos... attris) noexcept -> tl::expected<void, std::string> {
	static_assert(sizeof...(attris) > 0, "must have at least one attribute");

	GLBufferRef buf;
	TL_TRY_ASSIGN(buf, GLBuffer::create(GL_ARRAY_BUFFER));
	TL_CHECK_AND_PASS(buf->bind());
	{
		TL_CHECK_AND_PASS(buf->set_data(raw_data));

		auto set_attr = [](auto&& info) -> void {
			glEnableVertexAttribArray(info.slot);
			glVertexAttribPointer(info.slot, info.components, info.type, GL_FALSE, info.stride, 
				reinterpret_cast<const void*>(info.offset));
		};
		(set_attr(attris), ...);

		CHECK_GL_ERROR();
	}
	buf->unbind();
	CHECK_GL_ERROR();

	m_buffers.emplace_back(std::move(buf));

	return {};
}
