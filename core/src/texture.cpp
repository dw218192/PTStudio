#include "texture.h"
#include "stb_image_write.h"

using namespace PTS;

Texture::Texture(unsigned width, unsigned height, unsigned num_channel) noexcept :
	m_width{width}, m_height{height}, m_num_channel{ num_channel }, m_linear_sz{ width * height * num_channel }
{
    m_pixels.resize(m_linear_sz);
}

auto Texture::resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string> {
    m_width = width;
    m_height = height;
    m_linear_sz = width * height * 3;
    m_pixels.resize(m_linear_sz, 0);

    return {};
}

auto Texture::save(FileFormat fmt, std::string_view file_path) const noexcept -> tl::expected<void, std::string> {
    auto res = fetch_pixels();
    if (!res) {
        return tl::make_unexpected(res.error());
    }

    // stbi_flip_vertically_on_write(1);
    switch (fmt) {
        case FileFormat::PNG:
            stbi_write_png(file_path.data(), m_width, m_height, m_num_channel, m_pixels.data(), m_width * 3);
            break;
        case FileFormat::BMP:
            stbi_write_bmp(file_path.data(), m_width, m_height, m_num_channel, m_pixels.data());
            break;
        case FileFormat::TGA:
            stbi_write_tga(file_path.data(), m_width, m_height, m_num_channel, m_pixels.data());
            break;
        case FileFormat::JPG:
            stbi_write_jpg(file_path.data(), m_width, m_height, m_num_channel, m_pixels.data(), 100);
            break;
    }

    return {};
}

void Texture::swap(Texture&& other) noexcept {
    std::swap(m_width, other.m_width);
    std::swap(m_height, other.m_height);
    std::swap(m_num_channel, other.m_num_channel);
    std::swap(m_linear_sz, other.m_linear_sz);
    m_pixels.swap(other.m_pixels);
}
