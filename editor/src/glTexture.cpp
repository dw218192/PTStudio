#include "include/glTexture.h"
#include "include/ext.h"

auto GLTexture::fetch_pixels() const noexcept -> tl::expected<void, std::string> {
    try {
        bind();
    } catch (std::bad_any_cast const&) {
        return tl::unexpected { "invalid handle" };
    }

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, m_pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    if (auto err = glGetError(); err != GL_NO_ERROR) {
        return unexpected_gl_error(err);
    }

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

void GLTexture::bind() const noexcept {
    glBindTexture(GL_TEXTURE_2D, std::any_cast<GLuint>(m_handle));
}

void GLTexture::unbind() const noexcept {
    glBindTexture(GL_TEXTURE_2D, 0);
}