#include "include/glTexture.h"
#include <fstream>

using namespace PTS;

namespace {
	auto get_num_channels(GLenum format) noexcept -> unsigned {
		switch (format) {
		case GL_RED:
		case GL_GREEN:
		case GL_BLUE:
		case GL_RED_INTEGER:
		case GL_GREEN_INTEGER:
		case GL_BLUE_INTEGER:
		case GL_ALPHA:
		case GL_LUMINANCE:
		case GL_LUMINANCE_ALPHA:
		case GL_STENCIL_INDEX:
		case GL_COLOR_INDEX:
		case GL_DEPTH_COMPONENT:
		case GL_DEPTH_STENCIL:
			return 1;
		case GL_RG:
		case GL_RG_INTEGER:
			return 2;
		case GL_RGB:
		case GL_BGR:
		case GL_RGB_INTEGER:
		case GL_BGR_INTEGER:
			return 3;
		case GL_RGBA:
		case GL_BGRA:
		case GL_RGBA_INTEGER:
		case GL_BGRA_INTEGER:
		case GL_RGBA8:
		case GL_RGBA8_SNORM:
			return 4;

		default:
			return 0;
		}
	}
}

auto GLTexture::fetch_pixels() const noexcept -> tl::expected<void, std::string> {
	TL_CHECK_AND_PASS(bind());
	glGetTexImage(GL_TEXTURE_2D, 0, m_format, GL_UNSIGNED_BYTE, m_pixels.data());
	CHECK_GL_ERROR();
	unbind();

	return {};
}

auto GLTexture::create_tex(unsigned width, unsigned height, GLenum format, unsigned char const* data,
                           std::initializer_list<GLParam> params) noexcept
	-> tl::expected<GLuint, std::string> {
	GLuint tex;
	glGenTextures(1, &tex);
	CHECK_GL_ERROR();

	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(
		GL_TEXTURE_2D, 0, format, // RGB color format
		width, height,
		0, format, GL_UNSIGNED_BYTE, data
	);
	CHECK_GL_ERROR();

	for (auto&& [pname, param] : params) {
		glTexParameteri(GL_TEXTURE_2D, pname, param);
		CHECK_GL_ERROR();
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	return tex;
}

auto GLTexture::create(unsigned width, unsigned height, GLenum format, std::initializer_list<GLParam> params) noexcept
	-> tl::expected<GLTextureRef, std::string> {
	GLuint tex;
	TL_TRY_ASSIGN(tex, create_tex(width, height, format, nullptr, params));
	auto num_channels = get_num_channels(format);
	if (!num_channels) {
		return TL_ERROR("unsupported image format");
	}
	auto ret = GLTextureRef{new GLTexture{width, height, num_channels, format, tex}, GLResourceDeleter{}};
	return ret;
}

auto GLTexture::create(std::string_view img_file, FileFormat type, std::initializer_list<GLParam> params) noexcept
	-> tl::expected<GLTextureRef, std::string> {
	std::ifstream file{img_file.data(), std::ios::binary};
	if (!file) {
		return TL_ERROR("failed to open file");
	}
	file.seekg(0, std::ios::end);
	auto const size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<unsigned char> img_mem(size);
	file.read(reinterpret_cast<char*>(img_mem.data()), size);
	return create(tcb::make_span(img_mem), type, params);
}

auto GLTexture::create(unsigned width, unsigned height, GLenum format, GLuint handle) noexcept
	-> tl::expected<GLTextureRef, std::string> {
	auto const channels = get_num_channels(format);
	if (!channels) {
		return TL_ERROR("unsupported image format");
	}
	auto ret = GLTextureRef{new GLTexture{width, height, channels, format, handle}, GLResourceDeleter{}};
	return ret;
}

GLTexture::GLTexture(unsigned width, unsigned height, unsigned channels, GLenum format, GLuint handle)
	: Texture(width, height, channels), GLResource(handle), m_format(format) {}

GLTexture::~GLTexture() noexcept {
	if (m_handle) {
		glDeleteTextures(1, &m_handle);
	}
}

GLTexture::GLTexture(GLTexture&& other) noexcept {
	swap(std::move(other));
}

GLTexture& GLTexture::operator=(GLTexture&& other) noexcept {
	swap(std::move(other));
	return *this;
}

auto GLTexture::bind() const noexcept -> tl::expected<void, std::string> {
	if (!m_handle) {
		return TL_ERROR("texture is not valid");
	}
	glBindTexture(GL_TEXTURE_2D, m_handle);
	CHECK_GL_ERROR();
	return {};
}

void GLTexture::unbind() const noexcept {
	glBindTexture(GL_TEXTURE_2D, 0);
}

inline void GLTexture::swap(GLTexture&& other) noexcept {
	if (this == &other) {
		return;
	}
	std::swap(m_format, other.m_format);
	this->Texture::swap(std::move(other));
	this->GLResource::swap(std::move(other)); // NOLINT(bugprone-use-after-move)
}

auto GLTexture::get_id() const noexcept -> void* {
	return reinterpret_cast<void*>(m_handle);
}

auto GLTexture::resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> {
	TL_CHECK_AND_PASS(Texture::resize(width, height));
	glTexImage2D(
		GL_TEXTURE_2D, 0, m_format, // RGB color format
		width, height,
		0, m_format, GL_UNSIGNED_BYTE, nullptr
	);
	m_width = width;
	m_height = height;
	return {};
}
