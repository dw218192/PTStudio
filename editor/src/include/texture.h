#pragma once
#include <string>
#include <vector>
#include <any>

#include <GL/glew.h>
#include <tl/expected.hpp>

struct Texture;

enum class FileFormat {
    PNG,
    BMP,
    JPG,
    TGA
};

using TextureRef = std::reference_wrapper<Texture const>;

struct Texture {
    Texture(unsigned width, unsigned height, std::any handle) noexcept;
    ~Texture() noexcept = default;

    [[nodiscard]] auto get_pixels() const noexcept -> std::vector<unsigned char> const& {
        return m_pixels;
    }
    [[nodiscard]] auto get_pixels() noexcept -> std::vector<unsigned char>& {
        return m_pixels;
    }
    /**
     * \brief Gets the width of the render result, in num of pixels.
     * \return The width of the render result, in num of pixels.
    */
    [[nodiscard]] auto get_width() const noexcept { return m_width; }
    /**
     * \brief Gets the height of the render result, in num of pixels.
     * \return The height of the render result, in num of pixels.
    */
    [[nodiscard]] auto get_height() const noexcept { return m_height; }

    [[nodiscard]] auto get_handle() const noexcept -> std::any const& { return m_handle; }

    /**
     * \brief Saves the render result to a file
     * \param fmt The file format to be used
     * \param file_path The path to the file to be saved
     * \return on failure, a Result object that contains an error message\n
     * on success, an empty Result object.
    */
    auto save(FileFormat fmt, std::string_view file_path) const noexcept -> tl::expected<void, std::string>;

private:
    mutable std::vector<unsigned char> m_pixels;
    unsigned m_width, m_height, m_linear_sz;
    std::any m_handle;
};