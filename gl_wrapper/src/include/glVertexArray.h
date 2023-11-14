#pragma once

#include <string>
#include <tcb/span.hpp>
#include <tl/expected.hpp>
#include <glm/glm.hpp>

#include "glBuffer.h"
#include "glResource.h"

namespace PTS {
	struct GLVertexArray;
	using GLVertexArrayRef = UniqueGLResRef<GLVertexArray>;

	struct GLVertexArray final : GLResource {
		template <typename VertexType, std::size_t Extent, typename... GLAttributeInfos>
		[[nodiscard]] static auto create(tcb::span<VertexType const, Extent> vertex_data, GLAttributeInfos... attris)
			-> tl::expected<GLVertexArrayRef, std::string>;
		
		template <typename VertexType, std::size_t Extent0, std::size_t Extent1, typename... GLAttributeInfos>
		[[nodiscard]] static auto create_indexed(
			tcb::span<VertexType const, Extent0> vertex_data,
			tcb::span<unsigned const, Extent1> index_data,
			GLAttributeInfos... attris
		) -> tl::expected<GLVertexArrayRef, std::string>;

		GLVertexArray(GLVertexArray const&) = delete;
		auto operator=(GLVertexArray const&) -> GLVertexArray& = delete;
		GLVertexArray(GLVertexArray&& other) noexcept;
		auto operator=(GLVertexArray&& other) noexcept -> GLVertexArray&;

		[[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
		static void unbind() noexcept;
		[[nodiscard]] auto num_vertices() const noexcept -> GLsizei { 
			return m_num_vertices;
		}
		/**
		 * @brief Draw the vertex array, using index buffer if it is set.
		*/
		[[nodiscard]] auto draw(GLenum mode) const noexcept -> tl::expected<void, std::string>;

	private:
		/**
		 * @brief Specifies the layout of the vertex data.
		*/
		template <typename VertexType, std::size_t Extent, typename... GLAttributeInfos>
		[[nodiscard]] auto connect(tcb::span<VertexType const, Extent> raw_data,
		             GLAttributeInfos... attris) noexcept -> tl::expected<void, std::string>;

		void swap(GLVertexArray&& other) noexcept;
		GLVertexArray(GLuint handle, GLsizei num_vertices) noexcept;
		~GLVertexArray() noexcept override;

		GLsizei m_num_vertices {0};
		std::vector<GLBufferRef> m_buffers;
		GLBufferRef m_element_buf { nullptr };
	};

	struct GLAttributeInfoBase {
		GLuint slot, stride, offset;

		GLAttributeInfoBase(GLuint slot, GLuint stride, GLuint offset) : slot(slot), stride(stride), offset(offset) {}
	};

	template <typename T>
	struct GLAttributeInfo {
	};

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

	template <typename VertexType, std::size_t Extent, typename ... GLAttributeInfos>
	auto GLVertexArray::create(tcb::span<VertexType const, Extent> vertex_data,
		GLAttributeInfos... attris) -> tl::expected<GLVertexArrayRef, std::string> {
			auto vao = GLuint{};
			glGenVertexArrays(1, &vao);
			CHECK_GL_ERROR();

			auto num_vert = static_cast<GLsizei>(vertex_data.size());
			auto ret = GLVertexArrayRef{ new GLVertexArray { vao, num_vert }, GLResourceDeleter{} };
			
			glBindVertexArray(vao);
			CHECK_GL_ERROR();
			{
				TL_CHECK(ret->connect(vertex_data, attris...));
			}
			glBindVertexArray(0);

			return std::move(ret);
		}

	template <typename VertexType, std::size_t Extent0, std::size_t Extent1, typename... GLAttributeInfos>
	auto GLVertexArray::create_indexed(
		tcb::span<VertexType const, Extent0> vertex_data,
		tcb::span<unsigned const, Extent1> index_data,
		GLAttributeInfos... attris
	) -> tl::expected<GLVertexArrayRef, std::string> {
		auto vao = GLuint{};
		glGenVertexArrays(1, &vao);
		CHECK_GL_ERROR();

		auto num_vert = static_cast<GLsizei>(index_data.size());
		auto ret = GLVertexArrayRef{ new GLVertexArray { vao, num_vert }, GLResourceDeleter{} };
		
		GLBufferRef element_buf;
		TL_TRY_ASSIGN(element_buf, GLBuffer::create(GL_ELEMENT_ARRAY_BUFFER));
		TL_CHECK(element_buf->bind());
		TL_CHECK(element_buf->set_data(index_data));
		element_buf->unbind();
		
		glBindVertexArray(vao);
		CHECK_GL_ERROR();
		{
			TL_CHECK(ret->connect(vertex_data, attris...));
			TL_CHECK(element_buf->bind());
		}
		glBindVertexArray(0);
		ret->m_element_buf = std::move(element_buf);
		return std::move(ret);
	}

	template <typename VertexType, std::size_t Extent, typename... GLAttributeInfos>
	auto GLVertexArray::connect(tcb::span<VertexType const, Extent> raw_data,
	                            GLAttributeInfos... attris) noexcept -> tl::expected<void, std::string> {
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
}
