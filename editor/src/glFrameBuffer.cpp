#include "include/glFrameBuffer.h"
#include "include/glRenderBuffer.h"
#include "include/glTexture.h"
#include "utils.h"

auto GLFrameBuffer::create() -> tl::expected<GLFrameBufferRef, std::string> {
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	CHECK_GL_ERROR();
	return GLFrameBufferRef{ new GLFrameBuffer{ fbo }, GLResourceDeleter{} };
}

auto GLFrameBuffer::bind() const noexcept -> tl::expected<void, std::string> {
	if (!m_handle) {
		return TL_ERROR("frame buffer is not valid");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, m_handle);
	CHECK_GL_ERROR();
	return {};
}

GLFrameBuffer::GLFrameBuffer(GLFrameBuffer&& other) noexcept {
	swap(std::move(other));
}
auto GLFrameBuffer::operator=(GLFrameBuffer&& other) noexcept -> GLFrameBuffer& {
	swap(std::move(other));
	return *this;
}

void GLFrameBuffer::unbind() noexcept {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

auto GLFrameBuffer::attach(unsigned width, unsigned height, std::initializer_list<AttachmentDesc> descs) -> tl::expected<void, std::string> {
	for (auto const& desc : descs) {
		if (desc.attachment == GL_DEPTH_ATTACHMENT) {
			GLRenderBufferRef rbo;
			TL_ASSIGN(rbo, GLRenderBuffer::create(width, height, desc.format));
			TL_CHECK_FWD(rbo->bind());
			{
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, desc.attachment, GL_RENDERBUFFER, rbo->handle());
				CHECK_GL_ERROR();
			}
			rbo->unbind();
			m_rbo_attchs[desc.attachment] = std::move(rbo);
		} else if (desc.attachment == GL_COLOR_ATTACHMENT0) {
			GLTextureRef tex;
			TL_ASSIGN(tex, GLTexture::create(width, height, desc.format));
			TL_CHECK_FWD(tex->bind());
			{
				glFramebufferTexture(GL_FRAMEBUFFER, desc.attachment, tex->handle(), 0);
				CHECK_GL_ERROR();
			}
			tex->unbind();
			m_tex_attchs[desc.attachment] = std::move(tex);
		} else {
			return TL_ERROR("unsupported attachment");
		}
	}
	return {};
}

auto GLFrameBuffer::set_draw_buffer(GLenum attachment) const -> tl::expected<void, std::string> {
	glDrawBuffers(1, &attachment);
	CHECK_GL_ERROR();
	return {};
}

auto GLFrameBuffer::resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> {
	for (auto&& [attachment, rbo] : m_rbo_attchs) {
		TL_CHECK_FWD(rbo->bind());
		{
			TL_CHECK_FWD(rbo->resize(width, height));
		}
		rbo->unbind();
	}
	for (auto&& [attachment, tex] : m_tex_attchs) {
		TL_CHECK_FWD(tex->bind());
		{
			TL_CHECK_FWD(tex->resize(width, height));
		}
		tex->unbind();
	}
	return {};
}

auto GLFrameBuffer::get_texture(GLenum attachment) const noexcept -> GLTexture const* {
	if (auto const it = m_tex_attchs.find(attachment); it != m_tex_attchs.end()) {
		return it->second.get();
	}
	return nullptr;
}

auto GLFrameBuffer::get_render_buffer(GLenum attachment) const noexcept -> GLRenderBuffer const* {
	if (auto const it = m_rbo_attchs.find(attachment); it != m_rbo_attchs.end()) {
		return it->second.get();
	}
	return nullptr;
}

auto GLFrameBuffer::swap_render_buffer(GLenum attachment, GLRenderBufferRef buf) noexcept -> tl::expected<GLRenderBufferRef, std::string> {
	if (auto const it = m_rbo_attchs.find(attachment); it != m_rbo_attchs.end()) {
		auto old = std::move(it->second);
		
		if (buf) {
			it->second = std::move(buf);
			TL_CHECK(buf->bind());
			{
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, buf->handle());
				CHECK_GL_ERROR();
			}
			buf->unbind();
		} else {
			m_rbo_attchs.erase(it);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, 0);
			CHECK_GL_ERROR();
		}
		
		return old;
	}
	return TL_ERROR("attachment not found");
}

auto GLFrameBuffer::swap_texture(GLenum attachment, GLTextureRef tex) noexcept -> tl::expected<GLTextureRef, std::string> {
	if (auto const it = m_tex_attchs.find(attachment); it != m_tex_attchs.end()) {
		auto old = std::move(it->second);
		
		if (tex) {
			TL_CHECK(tex->bind());
			{
				glFramebufferTexture(GL_FRAMEBUFFER, attachment, tex->handle(), 0);
				CHECK_GL_ERROR();
			}
			tex->unbind();
			it->second = std::move(tex);
		} else {
			glFramebufferTexture(GL_FRAMEBUFFER, attachment, 0, 0);
			CHECK_GL_ERROR();
			m_tex_attchs.erase(it);
		}
		
		return old;
	}
	return TL_ERROR("attachment not found");
}

void GLFrameBuffer::swap(GLFrameBuffer&& other) noexcept {
	if (this == &other) {
		return;
	}

	m_rbo_attchs.swap(other.m_rbo_attchs);
	m_tex_attchs.swap(other.m_tex_attchs);
	this->GLResource::swap(std::move(other));
}

GLFrameBuffer::GLFrameBuffer(GLuint handle) noexcept : GLResource(handle) {}

GLFrameBuffer::~GLFrameBuffer() noexcept {
	if (m_handle) {
		glDeleteFramebuffers(1, &m_handle);
	}
}