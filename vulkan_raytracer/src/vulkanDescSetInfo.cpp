#include "vulkanDescSetInfo.h"

auto PTS::VulkanDescSetInfo::add_binding(
    vk::DescriptorSetLayoutBinding binding,
    vk::DescriptorBindingFlags flags,
    vk::WriteDescriptorSet how_to_write
) -> VulkanDescSetInfo& {
    m_bindings.push_back(binding);
    m_flags.push_back(flags);
    m_writes.push_back(how_to_write);

    if (flags & vk::DescriptorBindingFlagBits::eUpdateAfterBind) {
        m_create_flags_pool = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
    }
    if (flags & vk::DescriptorBindingFlagBits::eVariableDescriptorCount) {
        m_variable_desc_count = true;
        m_max_desc_count = std::max(m_max_desc_count, binding.descriptorCount);
    }

    return *this;
}

[[nodiscard]] auto PTS::VulkanDescSetInfo::create(
    VulkanDeviceInfo const& dev,
    VulkanDescSetPoolInfo const& pool
) -> tl::expected<void, std::string> {
    try {
        auto flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo{}
            .setBindingCount(static_cast<uint32_t>(m_flags.size()))
            .setPBindingFlags(m_flags.data());
        
        auto desc_set_layout_info = vk::DescriptorSetLayoutCreateInfo{}
            .setBindings(m_bindings)
            .setPNext(&flags_info)
            .setFlags(m_create_flags_pool);

        m_layout = dev->createDescriptorSetLayoutUnique(
            desc_set_layout_info
        );
        
        auto alloc_info = vk::DescriptorSetAllocateInfo{
            *pool,
            *m_layout
        };
        if (m_variable_desc_count) {
            auto variable_count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo{}
                .setDescriptorSetCount(1)
                .setPDescriptorCounts(&m_max_desc_count);
            alloc_info.setPNext(&variable_count_info);
        }

        auto desc_sets = dev->allocateDescriptorSetsUnique(
            alloc_info
        );
        this->handle = std::move(desc_sets[0]);

        for (auto i = 0u; i < m_writes.size(); ++i) {
            m_writes[i].setDstSet(*handle)
                .setDstBinding(m_bindings[i].binding)
                .setDescriptorCount(m_bindings[i].descriptorCount)
                .setDescriptorType(m_bindings[i].descriptorType);
        }
        dev->updateDescriptorSets(m_writes, {});
        return {};
    }
    catch (vk::SystemError& err) {
        return TL_ERROR(err.what());
    }
}