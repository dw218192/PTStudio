#pragma once
#include "vulkanHelper.h"
#include "vulkanBufferInfo.h"
#include "scene.h"

#include "utils.h"

namespace PTS {
    struct VulkanAccelStructInfo : VulkanInfo<vk::UniqueAccelerationStructureKHR> {
        [[nodiscard]] static auto create(
            VulkanDeviceInfo const& dev,
            VulkanCmdPoolInfo const& cmd_pool,
            vk::AccelerationStructureBuildGeometryInfoKHR geom_build_info,
            uint32_t primitive_count,
            uint32_t max_primitive_count
        ) -> tl::expected<VulkanAccelStructInfo, std::string>;
        
        VulkanBufferInfo storage_mem {}; // mem to store accel struct
        VulkanBufferInfo scratch_mem {}; // mem to build accel struct        
        vk::BuildAccelerationStructureFlagsKHR flags {};
        std::vector<vk::AccelerationStructureGeometryKHR> geometries {};
        
        auto get_desc_info() const noexcept -> vk::WriteDescriptorSetAccelerationStructureKHR {
            return vk::WriteDescriptorSetAccelerationStructureKHR { *handle };
        }
    };

    struct VulkanBottomAccelStructInfo {
        [[nodiscard]] static auto create(
            VulkanDeviceInfo const& dev,
            VulkanCmdPoolInfo const& cmd_pool,
            Object const& obj
        ) -> tl::expected<VulkanBottomAccelStructInfo, std::string>;

        VulkanAccelStructInfo accel {};

        // mem to store geometry info
        VulkanBufferInfo vertex_mem {};
        VulkanBufferInfo index_mem {};
        // uvs, normals, materials, etc. are stored in the pipeline instead of here
    };

    struct VulkanTopAccelStructInfo {
        VulkanTopAccelStructInfo() = default;
        VulkanTopAccelStructInfo(VulkanTopAccelStructInfo&&) = default;
        VulkanTopAccelStructInfo& operator=(VulkanTopAccelStructInfo&&) = default;
        NO_COPY(VulkanTopAccelStructInfo);

        [[nodiscard]] static auto create(
            VulkanDeviceInfo const& dev,
            VulkanCmdPoolInfo const& cmd_pool
        ) -> tl::expected<VulkanTopAccelStructInfo, std::string>;
        
        /**
         * given a bottom accel struct, add an instance of it to the top accel struct
         * @note the bottom accel's ownership is transferred to the top accel
         * @param bottom_accel the bottom accel struct to add
         * @param transform the transform to apply to the bottom accel struct
        */
        [[nodiscard]] auto add_instance(
            VulkanBottomAccelStructInfo&& bottom_accel,
            glm::mat4 const& transform
        ) noexcept -> tl::expected<size_t, std::string>;

        /**
         * remove an instance from the top accel struct
         * @param idx the index of the instance to remove
         * @return void on success, error string on failure
        */
        [[nodiscard]] auto remove_instance(size_t idx) noexcept -> tl::expected<void, std::string>;

        /**
         * update an instance's transform
         * @param idx the index of the instance to update
         * @param transform the new transform to apply to the instance
        */
        [[nodiscard]] auto update_instance_transform(size_t idx, glm::mat4 const& transform) noexcept -> tl::expected<void, std::string>;
        [[nodiscard]] auto get_accel() const -> auto const& {
            return m_accel;
        }

    private:
        [[nodiscard]] static auto to_mat4x3(glm::mat4 const& mat) noexcept -> vk::TransformMatrixKHR;

        /**
         * @brief Updates the instance data in GPU
         * @param dev the vulkan device
         * @param cmd_pool the vulkan command pool
         * @param idx the index of the instance to update
         * @param transform the new transform to apply to the instance
        */
        [[nodiscard]] auto update_instance_gpu(
            size_t idx,
            glm::mat4 const& transform
        ) noexcept -> tl::expected<void, std::string>;

        /**
         * @brief Updates the acceleration structure in GPU
         * @param dev the vulkan device
         * @param cmd_pool the vulkan command pool
         * @param build_type the build type
         * @return void on success, error string on failure
        */
        [[nodiscard]] auto update_accel_gpu(
            vk::BuildAccelerationStructureModeKHR build_type,
            size_t from, size_t to
        ) noexcept -> tl::expected<void, std::string>;

    private:
        VulkanTopAccelStructInfo(
            VulkanAccelStructInfo accel,
            VulkanBufferInfo ins_mem,
            std::vector<VulkanBottomAccelStructInfo> bottom_accels,
            std::vector<vk::AccelerationStructureInstanceKHR> instances,
            VulkanDeviceInfo const& dev,
            VulkanCmdPoolInfo const& cmd_pool
        ) {
            m_accel = std::move(accel);
            m_ins_mem = std::move(ins_mem);
            m_bottom_accels = std::move(bottom_accels);
            m_instances = std::move(instances);
            m_dev = &dev;
            m_cmd_pool = &cmd_pool;
        }

        VulkanAccelStructInfo m_accel {};
        // mem to store bottom accel instances
        VulkanBufferInfo m_ins_mem {};
        // refs to bottom accels, the top accel owns the bottom accels
        std::vector<VulkanBottomAccelStructInfo> m_bottom_accels {};
        std::vector<vk::AccelerationStructureInstanceKHR> m_instances {};
        ViewPtr<VulkanDeviceInfo> m_dev { nullptr };
        ViewPtr<VulkanCmdPoolInfo> m_cmd_pool { nullptr };
        
        // free index
        std::vector<size_t> m_free_idx {};
    };
} // namespace PTS