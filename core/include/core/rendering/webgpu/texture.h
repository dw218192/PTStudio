#pragma once

#include <core/rendering/webgpu/webgpu.h>

namespace pts::webgpu {

class Texture {
   public:
    Texture() = default;
    explicit Texture(WGPUTexture texture);

    Texture(const Texture&) = delete;
    auto operator=(const Texture&) -> Texture& = delete;

    Texture(Texture&& other) noexcept;
    auto operator=(Texture&& other) noexcept -> Texture&;

    ~Texture();

    [[nodiscard]] auto is_valid() const noexcept -> bool;
    [[nodiscard]] auto handle() const noexcept -> WGPUTexture;

   private:
    WGPUTexture m_texture = nullptr;
};

}  // namespace pts::webgpu
