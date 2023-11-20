#pragma once

#include <tcb/span.hpp>
#include <tl/expected.hpp>
#include <string>

#include "glResource.h"
#include "utils.h"

namespace PTS {
	struct GLBuffer;
	using GLBufferRef = UniqueGLResRef<GLBuffer>;

	struct GLBuffer final : GLResource {
		NO_COPY(GLBuffer);
		[[nodiscard]] static auto create(GLenum target) -> tl::expected<GLBufferRef, std::string>;

		GLBuffer(GLBuffer&& other) noexcept;
		auto operator=(GLBuffer&& other) noexcept -> GLBuffer&;

		// TODO: might be helpful to store the element type as well
		template<typename T, size_t Extent>
		[[nodiscard]] auto set_data(tcb::span<T const, Extent> data, GLenum usage = GL_STATIC_DRAW) -> tl::expected<void, std::string>;
		template<typename T, size_t Extent>
		[[nodiscard]] auto set_data(tcb::span<T, Extent> data, GLenum usage = GL_STATIC_DRAW) -> tl::expected<void, std::string>;

		[[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
		void unbind() const noexcept;

		[[nodiscard]] auto size() const noexcept { return m_size; }
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
	template <typename T, size_t Extent>
	auto GLBuffer::set_data(tcb::span<T, Extent> data, GLenum usage) -> tl::expected<void, std::string> {
		if (data.empty()) {
			return {};
		}
		glBufferData(m_target, data.size() * sizeof(T), data.data(), usage);
		CHECK_GL_ERROR();

		m_size = data.size();
		return {};
	}

}
