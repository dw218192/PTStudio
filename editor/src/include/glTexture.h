#include "texture.h"

struct GLTexture : Texture {
    using Texture::Texture;
    void bind() const noexcept override;
    void unbind() const noexcept override;
protected:
    NODISCARD auto fetch_pixels() const noexcept -> tl::expected<void, std::string> override;
};