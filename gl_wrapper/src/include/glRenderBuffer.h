#pragma once
#include "glResource.h"
#include <string>
#include <tl/expected.hpp>

namespace PTS {
	struct GLRenderBuffer;
	using GLRenderBufferRef = UniqueGLResRef<GLRenderBuffer>;

	struct GLRenderBuffer final : GLResource {
		[[nodiscard]] static auto create(unsigned width, unsigned height,
		                                 GLenum format) -> tl::expected<GLRenderBufferRef, std::string>;

		GLRenderBuffer(GLRenderBuffer const&) = delete;
		auto operator=(GLRenderBuffer const&) -> GLRenderBuffer& = delete;

		GLRenderBuffer(GLRenderBuffer&& other) noexcept;
		auto operator=(GLRenderBuffer&& other) noexcept -> GLRenderBuffer&;

		[[nodiscard]] auto resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string>;
		[[nodiscard]] auto bind() const noexcept -> tl::expected<void, std::string>;
		static void unbind() noexcept;

		[[nodiscard]] auto get_format() const noexcept -> GLenum { return m_format; }
		[[nodiscard]] auto get_width() const noexcept -> unsigned { return m_width; }
		[[nodiscard]] auto get_height() const noexcept -> unsigned { return m_height; }

	private:
		[[nodiscard]] static auto create_buf(unsigned width, unsigned height,
		                                     GLenum format) noexcept -> tl::expected<GLuint, std::string>;
		void swap(GLRenderBuffer&& other) noexcept;
		GLRenderBuffer(unsigned width, unsigned height, GLenum format, GLuint handle) noexcept;
		~GLRenderBuffer() noexcept override;

		unsigned m_width, m_height;
		GLenum m_format;
	};
}
