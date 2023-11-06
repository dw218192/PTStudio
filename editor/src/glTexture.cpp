#include "include/glTexture.h"
#include <fstream>
#include <stb_image.h>

using namespace PTS;

auto GLTexture::fetch_pixels() const noexcept -> tl::expected<void, std::string> {
    TL_CHECK_AND_PASS(bind());
    glGetTexImage(GL_TEXTURE_2D, 0, m_format, GL_UNSIGNED_BYTE, m_pixels.data());
    CHECK_GL_ERROR();
    unbind();

    return {};
}

auto GLTexture::create_tex(unsigned width, unsigned height, GLenum format, unsigned char const* data, std::initializer_list<GLParam> params) noexcept
-> tl::expected<GLuint, std::string>
{
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
-> tl::expected<GLTextureRef, std::string> 
{
    GLuint tex;
    TL_TRY_ASSIGN(tex, create_tex(width, height, format, nullptr, params));
    unsigned num_channels;
    switch (format) {
    case GL_RED: num_channels = 1; break;
    case GL_RG: num_channels = 2; break;
    case GL_RGB: num_channels = 3; break;
    case GL_RGBA: num_channels = 4; break;
    default: return TL_ERROR("unsupported image format");
    }
    auto ret = GLTextureRef{ new GLTexture{ width, height, num_channels, format, tex }, GLResourceDeleter{} };
    return ret;
}

auto GLTexture::create(tcb::span<unsigned char const> img_mem, FileFormat type, std::initializer_list<GLParam> params) noexcept
-> tl::expected<GLTextureRef, std::string>
{
    unsigned width, height, num_channels;
    auto data = stbi_load_from_memory(img_mem.data(), img_mem.size(),
                                      reinterpret_cast<int*>(&width), 
                                      reinterpret_cast<int*>(&height),
                                      reinterpret_cast<int*>(&num_channels), 0);
    if (!data) {
        return TL_ERROR("failed to load image");
    }

    auto w = width, h = height;
    // flip pixels vertically, because OpenGL expects the first pixel to be at the bottom left corner
    for (decltype(w) y = 0; y < h / 2; ++y) {
        for (decltype(w) x = 0; x < w; ++x) {
            int const frm = y * w + x;
            int const to = (h - y - 1) * w + x;
            for (int k = 0; k < num_channels; ++k) {
                std::swap(data[num_channels * frm + k], data[num_channels * to + k]);
            }
        }
    }

    GLenum format;
    switch (num_channels) {
    case 1: format = GL_RED; break;
    case 2: format = GL_RG; break;
    case 3: format = GL_RGB; break;
    case 4: format = GL_RGBA; break;
    default: return TL_ERROR("unsupported image format");
    }
    GLuint tex;
    TL_TRY_ASSIGN(tex, create_tex(width, height, format, data, params));

    stbi_image_free(data);
    auto ret = GLTextureRef{ new GLTexture{ width, height, num_channels, format, tex }, GLResourceDeleter{} };
    return ret;
}

auto GLTexture::create(std::string_view img_file, FileFormat type, std::initializer_list<GLParam> params) noexcept
-> tl::expected<GLTextureRef, std::string>
{
    std::ifstream file{ img_file.data(), std::ios::binary };
    if (!file) {
        return TL_ERROR("failed to open file");
    }
    file.seekg(0, std::ios::end);
    auto const size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> img_mem(size);
    file.read(reinterpret_cast<char*>(img_mem.data()), size);
    return create(img_mem, type, params);
}

GLTexture::GLTexture(unsigned width, unsigned height, unsigned channels, GLenum format, GLuint handle)
	: Texture(width, height, channels), GLResource(handle), m_format(format) {}

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
