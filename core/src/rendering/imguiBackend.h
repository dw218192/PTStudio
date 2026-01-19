#pragma once

#include <core/loggingManager.h>
#include <imgui.h>

#include <memory>

#include "present.h"
#include "vulkan/vulkanRhi.h"

namespace pts::rendering {

class ImGuiBackend {
   public:
    ImGuiBackend(IWindowing& windowing, IPresent& present, VulkanRhi& rhi,
                 LoggingManager& logging_manager);
    ~ImGuiBackend();

    ImGuiBackend(const ImGuiBackend&) = delete;
    ImGuiBackend& operator=(const ImGuiBackend&) = delete;
    ImGuiBackend(ImGuiBackend&&) = delete;
    ImGuiBackend& operator=(ImGuiBackend&&) = delete;

    void new_frame();
    void render(bool framebuffer_resized);
    void resize();

    [[nodiscard]] auto register_texture(vk::Sampler sampler, vk::ImageView view,
                                        vk::ImageLayout layout) -> ImTextureID;
    void unregister_texture(ImTextureID id);

    struct Impl;

   private:
    std::unique_ptr<Impl> m_impl;
};

}  // namespace pts::rendering
