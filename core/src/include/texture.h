#pragma once
#include "utils.h"

#include <string>
#include <vector>

#include <tl/expected.hpp>
namespace PTS {
    struct Texture;

    enum class FileFormat {
        PNG,
        BMP,
        JPG,
        TGA
    };

    using TextureHandle = Texture const*;

    struct Texture {
        Texture(unsigned width, unsigned height, unsigned num_channel) noexcept;
        virtual ~Texture() noexcept = default;

        NODISCARD auto get_pixels() const noexcept -> std::vector<unsigned char> const& {
            return m_pixels;
        }
        NODISCARD auto get_pixels() noexcept -> std::vector<unsigned char>& {
            return m_pixels;
        }
        /**
         * \brief Gets the width of the render result, in num of pixels.
         * \return The width of the render result, in num of pixels.
        */
        NODISCARD auto get_width() const noexcept { return m_width; }
        /**
         * \brief Gets the height of the render result, in num of pixels.
         * \return The height of the render result, in num of pixels.
        */
        NODISCARD auto get_height() const noexcept { return m_height; }

        /**
         * \brief Changes the height of the texture
         * \param width the new width
         * \param height the new height
         * \return on failure, an error message\n
         */
        NODISCARD virtual auto resize(unsigned width, unsigned height) noexcept -> tl::expected<void, std::string>;

        /**
         * \brief Saves the render result to a file
         * \param fmt The file format to be used
         * \param file_path The path to the file to be saved
         * \return on failure, an error message
        */
        auto save(FileFormat fmt, std::string_view file_path) const noexcept -> tl::expected<void, std::string>;

        virtual auto bind() const noexcept -> tl::expected<void, std::string> = 0;
        virtual void unbind() const noexcept = 0;
        virtual auto get_id() const noexcept -> void* = 0;

    protected:
        Texture() noexcept = default;
        void swap(Texture&& other) noexcept;

        NODISCARD virtual auto fetch_pixels() const noexcept -> tl::expected<void, std::string> = 0;

        mutable std::vector<unsigned char> m_pixels;
        unsigned m_width{ 0 };
        unsigned m_height{ 0 };
        unsigned m_num_channel{ 0 };
        unsigned m_linear_sz{ 0 };
    };
}