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
    friend struct Renderer;
    RenderResult() noexcept;
    ~RenderResult() noexcept;
	RenderResult(RenderResult&) = delete;
    RenderResult(RenderResult&&) = delete;
    RenderResult& operator=(RenderResult&) = delete;
    RenderResult& operator=(RenderResult&&) = delete;

    [[nodiscard]] auto init(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto get_pixels() const noexcept -> std::vector<unsigned char> const& {
        return m_pixels;
    }

    /**
     * \brief Gets the width of the render result, in num of pixels.
     * \return The width of the render result, in num of pixels.
    */
    [[nodiscard]] auto get_width() const noexcept -> int { return m_width; }
    /**
     * \brief Gets the height of the render result, in num of pixels.
     * \return The height of the render result, in num of pixels.
    */
    [[nodiscard]] auto get_height() const noexcept -> int { return m_height; }

    /**
     * \brief Saves the render result to a file
     * \param fmt The file format to be used
     * \param file_path The path to the file to be saved
     * \return on failure, a Result object that contains an error message\n
     * on success, an empty Result object.
    */
    void save(FileFormat fmt, std::string_view file_path) const noexcept;
    /**
     * \brief Uploads the render result to the frame buffer.\n
     * Call this function if you want to display the render result on the screen.
     * \return on failure, a Result object that contains an error message\n
     * on success, an empty Result object.
    */
    auto upload_to_frame_buffer() const noexcept -> tl::expected<void, std::string>;
private:
    void prepare() const noexcept;
    void update() noexcept;

    GLuint m_fbo;
    GLuint m_tex;
    GLuint m_depth_rbo;
    
    std::vector<unsigned char> m_pixels;
    GLsizei m_width, m_height, m_linear_sz;
};