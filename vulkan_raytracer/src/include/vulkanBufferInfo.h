#pragma once
#include "utils.h"
#include "vulkanHelper.h"

namespace PTS {
struct VulkanBufferInfo : VulkanInfo<vk::UniqueBuffer> {
    enum class Type {
        Scratch,       // for building an accel struct
        Uniform,       // UBO
        Storage,       // SSBO
        AccelInput,    // for data stored by the accel struct, e.g. geometry info like vertices and
                       // indices or instances
        AccelStorage,  // for storing the top level accel struct
        ShaderBindingTable,  // for storing the shader binding table
    };

    NO_COPY(VulkanBufferInfo);
    VulkanBufferInfo() = default;
    VulkanBufferInfo(VulkanBufferInfo&&) = default;
    VulkanBufferInfo& operator=(VulkanBufferInfo&&) = default;

    [[nodiscard]] auto get_desc_info() const noexcept -> vk::DescriptorBufferInfo {
        return vk::DescriptorBufferInfo{*handle, 0, VK_WHOLE_SIZE};
    }

    [[nodiscard]] auto get_device_addr() const noexcept -> vk::DeviceAddress {
        return m_device_addr;
    }
    [[nodiscard]] auto get_mem() const noexcept -> auto const& {
        return m_mem;
    }
    [[nodiscard]] auto get_size() const noexcept {
        return m_size_bytes;
    }

    /**
     * @brief Convenience function to create a vulkan buffer
     * @param dev The vulkan device
     * @param type The type of the buffer
     * @param size The size of the buffer in bytes
     *
     * @param data The data to be copied to the buffer
     * @return A vulkan buffer
     */
    template <typename ElementType = int, std::size_t Extent = 0u>
    [[nodiscard]] static auto create(VulkanDeviceInfo const& dev, VulkanBufferInfo::Type type,
                                     vk::DeviceSize size, tcb::span<ElementType, Extent> data = {},
                                     size_t offset_bytes = 0)
        -> tl::expected<VulkanBufferInfo, std::string>;

    /**
     * @brief Upload data to the buffer immediately
     * @param data The data to be copied to the buffer
     * @param offset_bytes The offset in bytes to start copying to
     * @return void on success, error message on failure
     */
    template <typename ElementType = int, std::size_t Extent = 0u>
    [[nodiscard]] auto upload(tcb::span<ElementType, Extent> data, size_t offset_bytes = 0)
        -> tl::expected<void, std::string>;

    template <typename ElementType>
    [[nodiscard]] auto upload(ElementType const& data, size_t offset_bytes = 0)
        -> tl::expected<void, std::string> {
        return upload(tcb::span{&data, 1}, offset_bytes);
    }
    template <typename ElementType>
    [[nodiscard]] auto upload(ElementType& data, size_t offset_bytes = 0)
        -> tl::expected<void, std::string> {
        return upload(tcb::span{&data, 1}, offset_bytes);
    }
    // disallow temporary objects
    template <typename ElementType>
    [[nodiscard]] auto upload(ElementType&&, size_t) = delete;
    template <typename ElementType>
    [[nodiscard]] auto upload(ElementType const&&, size_t) = delete;

   private:
    VulkanBufferInfo(vk::UniqueBuffer buffer, vk::UniqueDeviceMemory mem, size_t size_bytes,
                     vk::DeviceAddress device_addr, VulkanDeviceInfo const& dev) noexcept
        : VulkanInfo{std::move(buffer)},
          m_mem{std::move(mem)},
          m_size_bytes{size_bytes},
          m_device_addr{device_addr},
          m_dev{&dev} {
    }

    vk::UniqueDeviceMemory m_mem{};
    size_t m_size_bytes{};
    vk::DeviceAddress m_device_addr{};
    ViewPtr<VulkanDeviceInfo> m_dev{nullptr};
};

template <typename ElementType, std::size_t Extent>
[[nodiscard]] auto PTS::VulkanBufferInfo::create(VulkanDeviceInfo const& dev,
                                                 VulkanBufferInfo::Type type, vk::DeviceSize size,
                                                 tcb::span<ElementType, Extent> data,
                                                 size_t offset_bytes)
    -> tl::expected<VulkanBufferInfo, std::string> {
    if (!size) {
        return TL_ERROR("buffer size must be greater than 0");
    }

    try {
        auto usage_flags = vk::BufferUsageFlags{};
        auto mem_props_flags = vk::MemoryPropertyFlags{};
        switch (type) {
            case VulkanBufferInfo::Type::Scratch:
                usage_flags = vk::BufferUsageFlagBits::eStorageBuffer;
                mem_props_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
                break;
            case VulkanBufferInfo::Type::Uniform:
                usage_flags =
                    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst;
                mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent;
                break;
            case VulkanBufferInfo::Type::Storage:
                usage_flags =
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
                mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent;
                break;
            case VulkanBufferInfo::Type::AccelInput:
                usage_flags = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                              vk::BufferUsageFlagBits::eStorageBuffer |
                              vk::BufferUsageFlagBits::eTransferDst;
                mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent;
                break;
            case VulkanBufferInfo::Type::AccelStorage:
                usage_flags = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR;
                mem_props_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
                break;
            case VulkanBufferInfo::Type::ShaderBindingTable:
                usage_flags = vk::BufferUsageFlagBits::eShaderBindingTableKHR;
                mem_props_flags = vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent;
                break;
            default:
                return TL_ERROR("invalid buffer type");
        }
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;

        auto buffer = dev->createBufferUnique(vk::BufferCreateInfo{}
                                                  .setSize(size)
                                                  .setUsage(usage_flags)
                                                  .setSharingMode(vk::SharingMode::eExclusive));
        // get property index
        auto const mem_req = dev->getBufferMemoryRequirements(*buffer);
        auto mem_type_idx = std::numeric_limits<uint32_t>::max();
        auto mem_props = dev.physical_device.getMemoryProperties();
        for (auto i = 0u; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_req.memoryTypeBits & (1 << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & mem_props_flags) == mem_props_flags) {
                mem_type_idx = i;
                break;
            }
        }

        // TODO: is this necessary?
        size = mem_req.size;

        auto flags_info = vk::MemoryAllocateFlagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress};
        auto alloc_info = vk::MemoryAllocateInfo{mem_req.size, mem_type_idx};

        alloc_info.setPNext(&flags_info);
        auto mem = dev->allocateMemoryUnique(alloc_info);
        dev->bindBufferMemory(*buffer, *mem, 0);
        auto device_addr = dev->getBufferAddressKHR(vk::BufferDeviceAddressInfo{*buffer});
        auto ret = VulkanBufferInfo{std::move(buffer), std::move(mem), size, device_addr, dev};
        if (!data.empty()) {
            TL_CHECK(ret.upload(data, offset_bytes));
        }
        return ret;
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}

template <typename ElementType, std::size_t Extent>
auto VulkanBufferInfo::upload(tcb::span<ElementType, Extent> data, size_t offset_bytes)
    -> tl::expected<void, std::string> {
    if (!m_dev || !*m_dev) {
        return TL_ERROR("device is not valid");
    }

    if (offset_bytes > m_size_bytes) {
        return TL_ERROR("offset is larger than buffer size");
    }

    auto const byte_size = data.size() * sizeof(ElementType);
    if (byte_size > m_size_bytes - offset_bytes) {
        return TL_ERROR("data size is larger than buffer size");
    }

    auto& dev = *m_dev;
    try {
        // the size argument of mapMemory is actually the size of the mapped memory
        // not the size of the data to be copied, so VK_WHOLE_SIZE is used here instead of byte_size
        auto const mapped = dev->mapMemory(*m_mem, offset_bytes, VK_WHOLE_SIZE);
        memcpy(mapped, data.data(), byte_size);
        dev->unmapMemory(*m_mem);
        return {};
    } catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}
}  // namespace PTS