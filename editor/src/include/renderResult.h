#pragma once
#include <string>
#include <GL/glew.h>
#include <vector>
#include <tl/expected.hpp>

enum class FileFormat {
    PNG,
    BMP,
    JPG,
    TGA
};

/** \brief Represents the result of a render operation\n
 * This class is not meant to be created directly. Instead, use Renderer::render() to get a RenderResult object.
*/
struct RenderResult {
    RenderResult(unsigned width, unsigned height) noexcept;
    ~RenderResult() noexcept;
	RenderResult(RenderResult&) = delete;
    RenderResult(RenderResult&&) = delete;
    RenderResult& operator=(RenderResult&) = delete;
    RenderResult& operator=(RenderResult&&) = delete;

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

    /**
     * \brief Saves the render result to a file
     * \param fmt The file format to be used
     * \param file_path The path to the file to be saved
     * \return on failure, a Result object that contains an error message\n
     * on success, an empty Result object.
    */
    void save(FileFormat fmt, std::string_view file_path) const noexcept;

private:
    std::vector<unsigned char> m_pixels;
    unsigned m_width, m_height, m_linear_sz;
};