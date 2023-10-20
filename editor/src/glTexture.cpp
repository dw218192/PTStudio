#include "include/glTexture.h"
#include "include/ext.h"

auto GLTexture::fetch_pixels() const noexcept -> tl::expected<void, std::string> {
    bind();

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, m_pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    CHECK_GL_ERROR();

    auto w = m_width, h = m_height;
    // flip vertically
    for (decltype(w) y = 0; y < h / 2; ++y) {
        for (decltype(w) x = 0; x < w; ++x) {
            int const frm = y * w + x;
            int const to = (h - y - 1) * w + x;
            for (int k = 0; k < 3; ++k) {
                std::swap(m_pixels[3 * frm + k], m_pixels[3 * to + k]);
            }
        }
    }

    unbind();

    return {};
}

auto GLTexture::create_tex(unsigned width, unsigned height, GLenum format) noexcept -> tl::expected<GLuint, std::string> {
    GLuint tex;
    glGenTextures(1, &tex);
    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(
        GL_TEXTURE_2D, 0, format, // RGB color format
        width, height,
        0, format, GL_UNSIGNED_BYTE, nullptr
    );

    CHECK_GL_ERROR();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, 0);

    return tex;
}

auto GLTexture::create(unsigned width, unsigned height, GLenum format) -> tl::expected<GLTextureRef, std::string> {
    GLuint tex;
    TL_ASSIGN(tex, create_tex(width, height, format));
    auto ret = GLTextureRef{ new GLTexture{width, height, tex, format}, GLResourceDeleter{} };
    return ret;
}

GLTexture::GLTexture(unsigned width, unsigned height, GLuint handle, GLenum format)
	: Texture(width, height), GLResource(handle), m_format(format) {}

GLTexture::~GLTexture() noexcept {
    if(m_handle) {
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
    this->GLResource::swap(std::move(other));  // NOLINT(bugprone-use-after-move)
}

auto GLTexture::get_id() const noexcept -> void* {
    return reinterpret_cast<void*>(m_handle);
}

auto GLTexture::resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> {
	TL_CHECK_FWD(Texture::resize(width, height));
    glTexImage2D(
        GL_TEXTURE_2D, 0, m_format, // RGB color format
        width, height,
        0, m_format, GL_UNSIGNED_BYTE, nullptr
    );
    m_width = width;
    m_height = height;
	return {};
}
