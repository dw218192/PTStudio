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

auto GLTexture::create_tex(unsigned width, unsigned height) noexcept -> tl::expected<GLuint, std::string> {
    GLuint tex;
    glGenTextures(1, &tex);
    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, // RGB color format
        width, height,
        0, GL_RGB, GL_UNSIGNED_BYTE, nullptr
    );

    CHECK_GL_ERROR();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, 0);

    return tex;
}

auto GLTexture::create(unsigned width, unsigned height) -> tl::expected<GLTextureRef, std::string> {

    TL_CHECK(create_tex(width, height));
    auto tex = create_tex(width, height);
    if (!tex) return TL_ERROR(tex.error());

    auto ret = GLTextureRef{ new GLTexture{width, height}, GLResourceDeleter{} };
    ret->m_handle = tex.value();
    return ret;
}

GLTexture::GLTexture(unsigned width, unsigned height) : Texture{ width, height }, GLResource{} {}

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

inline void GLTexture::swap(GLTexture&& other) noexcept {
    this->Texture::swap(std::move(other));
    this->GLResource::swap(std::move(other));  // NOLINT(bugprone-use-after-move)
}

void GLTexture::bind() const noexcept {
    glBindTexture(GL_TEXTURE_2D, m_handle);
}

void GLTexture::unbind() const noexcept {
    glBindTexture(GL_TEXTURE_2D, 0);
}

auto GLTexture::get_handle() const noexcept -> void* {
    return reinterpret_cast<void*>(m_handle);
}

auto GLTexture::resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> {
	TL_CHECK_FWD(Texture::resize(width, height));
    auto tex = create_tex(width, height);
    if (!tex) return TL_ERROR(tex.error());

    m_handle = tex.value();
    return {};
}
