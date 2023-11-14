#include "vulkanGLInterop.h"

#include <iostream>

#include "utils.h"

auto PTS::VulkanGLInteropUtils::to_gl_fmt(vk::Format fmt) noexcept -> std::optional<GLenum> {
    // TODO: add more formats if needed
    switch (fmt) {
        // single component
        case vk::Format::eR8Unorm:
        case vk::Format::eR8Snorm:
        case vk::Format::eR8Uscaled:
        case vk::Format::eR8Sscaled:
        case vk::Format::eR8Uint:
        case vk::Format::eR8Sint:
        case vk::Format::eR8Srgb:
        case vk::Format::eR16Unorm:
        case vk::Format::eR16Snorm:
        case vk::Format::eR16Uscaled:
        case vk::Format::eR16Sscaled:
        case vk::Format::eR16Uint:
        case vk::Format::eR16Sint:
        case vk::Format::eR16Sfloat:
            return GL_RED;
        
        // two components
        case vk::Format::eR8G8Unorm:
        case vk::Format::eR8G8Snorm:
        case vk::Format::eR8G8Uscaled:
        case vk::Format::eR8G8Sscaled:
        case vk::Format::eR8G8Uint:
        case vk::Format::eR8G8Sint:
        case vk::Format::eR8G8Srgb:
        case vk::Format::eR16G16Unorm:
        case vk::Format::eR16G16Snorm:
        case vk::Format::eR16G16Uscaled:
        case vk::Format::eR16G16Sscaled:
        case vk::Format::eR16G16Uint:
        case vk::Format::eR16G16Sint:
        case vk::Format::eR16G16Sfloat:
            return GL_RG;
        
        // three components
        case vk::Format::eR8G8B8Unorm:
        case vk::Format::eR8G8B8Snorm:
        case vk::Format::eR8G8B8Uscaled:
        case vk::Format::eR8G8B8Sscaled:
        case vk::Format::eR8G8B8Uint:
        case vk::Format::eR8G8B8Sint:
        case vk::Format::eR8G8B8Srgb:
        case vk::Format::eB8G8R8Unorm:
        case vk::Format::eB8G8R8Snorm:
        case vk::Format::eB8G8R8Uscaled:
        case vk::Format::eB8G8R8Sscaled:
        case vk::Format::eB8G8R8Uint:
        case vk::Format::eB8G8R8Sint:
        case vk::Format::eB8G8R8Srgb:
            return GL_RGB;
        
        // four components
        case vk::Format::eR8G8B8A8Unorm:
            return GL_RGBA8;
        case vk::Format::eR8G8B8A8Snorm:
            return GL_RGBA8_SNORM;
        case vk::Format::eR8G8B8A8Uscaled:
        case vk::Format::eR8G8B8A8Sscaled:
        case vk::Format::eR8G8B8A8Uint:
        case vk::Format::eR8G8B8A8Sint:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eB8G8R8A8Unorm:
        case vk::Format::eB8G8R8A8Snorm:
        case vk::Format::eB8G8R8A8Uscaled:
        case vk::Format::eB8G8R8A8Sscaled:
        case vk::Format::eB8G8R8A8Uint:
        case vk::Format::eB8G8R8A8Sint:
        case vk::Format::eB8G8R8A8Srgb:
            return GL_RGBA;

        // depth formats
        // single component
        case vk::Format::eD16Unorm:
        case vk::Format::eX8D24UnormPack32:
        case vk::Format::eD32Sfloat:
            return GL_DEPTH_COMPONENT;
        // two components
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return GL_DEPTH_STENCIL;

        default:
            return std::nullopt; // No equivalent format
    }
}

PTS::VulkanGLInteropUtils::UniqueSystemHandle::UniqueSystemHandle(HandleType handle) noexcept 
    : handle { handle } {}
PTS::VulkanGLInteropUtils::UniqueSystemHandle::~UniqueSystemHandle() noexcept {
#if defined(_WIN32)
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
#elif defined(__linux__)
    if (handle != -1) {
        close(handle);
    }
#endif
}
PTS::VulkanGLInteropUtils::UniqueSystemHandle::UniqueSystemHandle(UniqueSystemHandle&& other) noexcept {
    std::swap(handle, other.handle);
}

auto PTS::VulkanGLInteropUtils::UniqueSystemHandle::operator=(UniqueSystemHandle&& other) noexcept
-> UniqueSystemHandle& {
    std::swap(handle, other.handle);
    return *this;
}

auto PTS::VulkanGLInteropUtils::UniqueSystemHandle::get() const noexcept -> HandleType {
    return handle;
}

PTS::VulkanGLInteropUtils::UniqueGLMemObject::UniqueGLMemObject(GLuint mem_obj) noexcept 
    : mem_obj{ mem_obj } {}

PTS::VulkanGLInteropUtils::UniqueGLMemObject::~UniqueGLMemObject() noexcept {
    if (mem_obj && glIsMemoryObjectEXT(mem_obj)) {
        glDeleteMemoryObjectsEXT(1, &mem_obj);
    }
}

PTS::VulkanGLInteropUtils::UniqueGLMemObject::UniqueGLMemObject(UniqueGLMemObject&& other) noexcept {
    std::swap(mem_obj, other.mem_obj);
}

auto PTS::VulkanGLInteropUtils::UniqueGLMemObject::operator=(UniqueGLMemObject&& other) noexcept
-> UniqueGLMemObject& {
    std::swap(mem_obj, other.mem_obj);
    return *this;
}
auto PTS::VulkanGLInteropUtils::UniqueGLMemObject::get() const noexcept -> GLuint {
    return mem_obj;
}

auto PTS::VulkanGLInteropUtils::get_vk_ins_exts() noexcept -> std::vector<std::string_view> {
    auto ret = std::vector<std::string_view> {};
    ret.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);

    return ret;
}

auto PTS::VulkanGLInteropUtils::get_vk_dev_exts() noexcept -> std::vector<std::string_view> {
    auto ret = std::vector<std::string_view>{};
    ret.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);

#if defined(_WIN32)
    ret.emplace_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#elif defined(__linux__)
    ret.emplace_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#else
    static_assert(false, "Unsupported platform");
#endif

    return ret;
}

auto PTS::VulkanGLInteropUtils::get_gl_exts() noexcept -> std::vector<std::string_view> {
    auto ret = std::vector<std::string_view> {};
    ret.emplace_back("GL_EXT_memory_object");

#if defined(_WIN32)
    ret.emplace_back("GL_EXT_memory_object_win32");
#elif defined(__linux__)
    ret.emplace_back("GL_EXT_memory_object_fd");
#else
    static_assert(false, "Unsupported platform");
#endif
    return ret;
}

auto PTS::VulkanGLInteropUtils::create_shared_image(
    vk::Device& dev,
    vk::ImageCreateInfo& img_info,
    vk::MemoryPropertyFlags mem_flags,
    vk::PhysicalDeviceMemoryProperties mem_info
) noexcept -> tl::expected<SharedImage, std::string> {
#if defined(_WIN32)
try{
    auto external_info = vk::ExternalMemoryImageCreateInfoKHR{}
        .setHandleTypes(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32);
    img_info.setPNext(&external_info);
    auto vk_img = dev.createImageUnique(img_info);
    if (!vk_img) {
        return TL_ERROR("Failed to create image");
    }

    auto const mem_req = dev.getImageMemoryRequirements(*vk_img);
    auto mem_type_idx = std::numeric_limits<uint32_t>::max();
    for (auto i = 0u; i < mem_info.memoryTypeCount; ++i) {
        if ((mem_req.memoryTypeBits & (1 << i)) && (mem_info.memoryTypes[i].propertyFlags & mem_flags) == mem_flags) {
            mem_type_idx = i;
            break;
        }
    }
    if (mem_type_idx == std::numeric_limits<uint32_t>::max()) {
        return TL_ERROR("Failed to find suitable memory type");
    }

    // Allocate memory with export info
    auto alloc_info = vk::MemoryAllocateInfo{}
        .setAllocationSize(mem_req.size)
        .setMemoryTypeIndex(mem_type_idx);
    auto const export_info = vk::ExportMemoryAllocateInfoKHR {}
        .setHandleTypes(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32);
    alloc_info.setPNext(&export_info);
    auto vk_mem = dev.allocateMemoryUnique(alloc_info);
    if (!vk_mem) {
        return TL_ERROR("Failed to allocate memory");
    }

    // Export Vulkan memory to win32 handle
    auto const get_mem_info = vk::MemoryGetWin32HandleInfoKHR{}
		.setMemory(*vk_mem)
		.setHandleType(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32);
    auto const handle = dev.getMemoryWin32HandleKHR(get_mem_info);
    if(!handle) {
        return TL_ERROR("Failed to get memory");
    }

    // Import the win32 handle into GL
    auto gl_mem = GLuint { 0 };
    glCreateMemoryObjectsEXT(1, &gl_mem);
    CHECK_GL_ERROR();
    glImportMemoryWin32HandleEXT(gl_mem, alloc_info.allocationSize, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
    CHECK_GL_ERROR();

    auto gl_fmt = to_gl_fmt(img_info.format);
    if (!gl_fmt) {
        return TL_ERROR("No equivalent GL format");
    }
    
    // Create GL texture backed by the same memory
    auto tex = GLTextureRef {};
    auto const extent = img_info.extent;
    auto gl_tex_handle = GLuint { 0 };
    glCreateTextures(GL_TEXTURE_2D, 1, &gl_tex_handle);
    CHECK_GL_ERROR();

    TL_TRY_ASSIGN(tex, GLTexture::create(extent.width, extent.height, *gl_fmt, gl_tex_handle)); 
    TL_CHECK(tex->bind());
    {
        glTextureStorageMem2DEXT(tex->handle(), 1, *gl_fmt, extent.width, extent.height, gl_mem, 0);
        CHECK_GL_ERROR();
        
        // TODO: should be consistent with the sampler used with vulkan
        glTextureParameteri(tex->handle(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(tex->handle(), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        CHECK_GL_ERROR();

        glTextureParameteri(tex->handle(), GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(tex->handle(), GL_TEXTURE_WRAP_T, GL_REPEAT);
        CHECK_GL_ERROR();
    }
    tex->unbind();

    return SharedImage { 
        { std::move(vk_mem), UniqueGLMemObject { gl_mem }, UniqueSystemHandle { handle } }, 
        std::move(vk_img), std::move(tex)
    };
} catch (vk::SystemError const& err) {
    return TL_ERROR("Vulkan error: " + std::string { err.what() });
} catch (std::exception const& err) {
    return TL_ERROR("Error: " + std::string { err.what() });
}
#elif defined(__linux__)
    return TL_ERROR("Function Not Implemented");
#else
    return TL_ERROR("Unsupported platform");
#endif
}