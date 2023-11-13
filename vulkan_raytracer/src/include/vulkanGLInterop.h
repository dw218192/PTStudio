#pragma once
#include <GL/glew.h>
#include <vulkan/vulkan.hpp>
#include <optional>
#include <tl/expected.hpp>
#include <vector>
#include <string_view>

#include "glTexture.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#elif defined(__linux__)
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace PTS {
    namespace VulkanGLInteropUtils {
        auto to_gl_fmt(vk::Format fmt) noexcept -> std::optional<GLenum>;

#if defined(_WIN32)
        using HandleType = HANDLE;
#else
        using HandleType = int;
#endif
        static HandleType const k_invalid_handle = reinterpret_cast<HandleType>(-1);

        struct UniqueSystemHandle {
            NO_COPY(UniqueSystemHandle);
            UniqueSystemHandle() noexcept = default;
            explicit UniqueSystemHandle(HandleType handle) noexcept;
            ~UniqueSystemHandle() noexcept;
            UniqueSystemHandle(UniqueSystemHandle&& other) noexcept;
            auto operator=(UniqueSystemHandle&& other) noexcept -> UniqueSystemHandle&;
            auto get() const noexcept -> HandleType;
        private:
            HandleType handle { k_invalid_handle };
        };

        struct UniqueGLMemObject {
            NO_COPY(UniqueGLMemObject);
            UniqueGLMemObject() noexcept = default;
            explicit UniqueGLMemObject(GLuint mem_obj) noexcept;
            ~UniqueGLMemObject() noexcept;
            UniqueGLMemObject(UniqueGLMemObject&& other) noexcept;
            auto operator=(UniqueGLMemObject&& other) noexcept -> UniqueGLMemObject&;
            auto get() const noexcept -> GLuint;
        private:
            GLuint mem_obj{ 0 };
        };
        
        /**
         * @brief This structure contains vulkan and GL objects backed by the same GPU memory.
        */
        struct SharedMem {
            vk::UniqueDeviceMemory vk_mem{};
            UniqueGLMemObject gl_mem{};
            UniqueSystemHandle shared_handle{};
        };

        /**
         * @brief This structure contains a vulkan image allocated for use with GL (external memory).
        */
        struct SharedImage {
            SharedMem mem {};
            vk::UniqueImage vk_image {};
            GLTextureRef gl_tex {};
            auto is_shared() const noexcept -> bool {
                return mem.shared_handle.get() != k_invalid_handle;
            }
        };

        auto get_vk_ins_exts() noexcept -> std::vector<std::string_view>;
        auto get_vk_dev_exts() noexcept -> std::vector<std::string_view>;
        auto get_gl_exts() noexcept -> std::vector<std::string_view>;

        auto create_shared_image(
            vk::Device& dev,
            vk::ImageCreateInfo& img_info,
            vk::MemoryPropertyFlags mem_flags,
            vk::PhysicalDeviceMemoryProperties mem_info
        ) noexcept -> tl::expected<SharedImage, std::string>;
    }
}