#include <core/rendering/webgpu/texture.h>

namespace pts::webgpu {

Texture::Texture(WGPUTexture texture) : m_texture(texture) {
}

Texture::Texture(Texture&& other) noexcept : m_texture(other.m_texture) {
    other.m_texture = nullptr;
}

auto Texture::operator=(Texture&& other) noexcept -> Texture& {
    if (this != &other) {
        if (m_texture != nullptr) {
            wgpuTextureRelease(m_texture);
        }
        m_texture = other.m_texture;
        other.m_texture = nullptr;
    }
    return *this;
}

Texture::~Texture() {
    if (m_texture != nullptr) {
        wgpuTextureRelease(m_texture);
    }
}

auto Texture::is_valid() const noexcept -> bool {
    return m_texture != nullptr;
}

auto Texture::handle() const noexcept -> WGPUTexture {
    return m_texture;
}

}  // namespace pts::webgpu
