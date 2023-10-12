#include "include/texture.h"
#include "include/ext.h"


Texture::Texture(unsigned width, unsigned height, std::any handle) noexcept :
	m_width{width}, m_height{height}, m_linear_sz{width*height*3}, m_handle{handle} {
    m_pixels.resize(m_linear_sz);
}

auto Texture::save(FileFormat fmt, std::string_view file_path) const noexcept -> tl::expected<void, std::string> {
    try {
        glBindTexture(GL_TEXTURE_2D, std::any_cast<GLuint>(m_handle));
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
    } catch (std::bad_any_cast const& e) {
        return tl::unexpected{ e.what() };
    }

    // stbi_flip_vertically_on_write(1);
    switch (fmt) {
        case FileFormat::PNG:
            stbi_write_png(file_path.data(), m_width, m_height, 3, m_pixels.data(), m_width * 3);
            break;
        case FileFormat::BMP:
            stbi_write_bmp(file_path.data(), m_width, m_height, 3, m_pixels.data());
            break;
        case FileFormat::TGA:
            stbi_write_tga(file_path.data(), m_width, m_height, 3, m_pixels.data());
            break;
        case FileFormat::JPG:
            stbi_write_jpg(file_path.data(), m_width, m_height, 3, m_pixels.data(), 100);
            break;
    }

    return {};
}