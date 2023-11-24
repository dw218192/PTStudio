#pragma once
#include "vulkanHelper.h"

namespace PTS {
struct VulkanDescSetInfo : VulkanInfo<vk::UniqueDescriptorSet> {
    auto add_binding(
        vk::DescriptorSetLayoutBinding binding,
        vk::DescriptorBindingFlags flags,
        vk::WriteDescriptorSet how_to_write
    ) -> VulkanDescSetInfo&;

    [[nodiscard]] auto create(
        VulkanDeviceInfo const& dev,
        VulkanDescSetPoolInfo const& pool
    ) -> tl::expected<void, std::string>;

    [[nodiscard]] auto get_layout() const { return *m_layout; }
    
private:
    vk::UniqueDescriptorSetLayout m_layout {};
    std::vector<vk::DescriptorSetLayoutBinding> m_bindings;
    std::vector<vk::DescriptorBindingFlags> m_flags;
    std::vector<vk::WriteDescriptorSet> m_writes;
    vk::DescriptorSetLayoutCreateFlagBits m_create_flags_pool {};
    bool m_variable_desc_count { false };
    uint32_t m_max_desc_count { 0 }; // max desc count of any binding, for variable desc count feature
};

} // namespace PTS
