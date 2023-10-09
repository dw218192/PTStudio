#include "include/renderResult.h"
#include "include/ext.h"
#include "include/shader.h"

void RenderResult::save(FileFormat fmt, std::string_view file_path) const noexcept {
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
}
RenderResult::RenderResult(unsigned width, unsigned height) noexcept :
	m_width{width}, m_height{height}, m_linear_sz{width*height*3} {
    m_pixels.resize(m_linear_sz);
}

RenderResult::~RenderResult() noexcept = default;
