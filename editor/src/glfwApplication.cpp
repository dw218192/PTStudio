#include "glfwApplication.h"

#include <core/imgui/imhelper.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

using namespace PTS;

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// stubs for callbacks
namespace PTS {
static void click_func(GLFWwindow* window, int button, int action, int mods) {
    // auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    // check if ImGui is using the mouse
    // static_cast<void>(mods);
    // app->m_mouse_states[button] = action == GLFW_PRESS;
}
static void motion_func(GLFWwindow* window, double x, double y) {
}
static void scroll_func(GLFWwindow* window, double x, double y) {
    auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    app->m_mouse_scroll_delta = {x, y};
}
static void key_func(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    // static_cast<void>(scancode);
    // static_cast<void>(mods);
    // app->m_key_states[key] = action == GLFW_PRESS;
}
static void error_func(int error, const char* description) {
    std::cerr << "GLFW error: " << error << ": " << description << std::endl;
    std::exit(-1);
}
static void framebuffer_resize_func(GLFWwindow* window, int width, int height) {
    auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    app->m_vk_framebuffer_resized = true;
    static_cast<void>(width);
    static_cast<void>(height);
}
}  // namespace PTS

GLFWApplication::GLFWApplication(std::string_view name, unsigned width, unsigned height,
                                 float min_frame_time)
    : Application{name} {
    set_min_frame_time(min_frame_time);
    glfwSetErrorCallback(error_func);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        std::exit(-1);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    m_window = glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create window" << std::endl;
        std::exit(-1);
    }

    // set callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, click_func);
    glfwSetCursorPosCallback(m_window, motion_func);
    glfwSetScrollCallback(m_window, scroll_func);
    glfwSetKeyCallback(m_window, key_func);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_resize_func);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    create_vulkan_instance();
    create_vulkan_surface();
}

GLFWApplication::~GLFWApplication() {
    shutdown_imgui_vulkan();
    if (m_vk_surface != VK_NULL_HANDLE && m_vk_instance) {
        m_vk_instance->destroySurfaceKHR(m_vk_surface);
        m_vk_surface = VK_NULL_HANDLE;
    }
    ImGui::DestroyContext();

    glfwTerminate();
}

void GLFWApplication::run() {
    static bool s_once = false;
    double last_frame_time = 0;
    while (!glfwWindowShouldClose(m_window)) {
        auto const now = glfwGetTime();

        // Poll and handle events (inputs, window resize, etc.)
        m_mouse_scroll_delta = glm::vec2{0.0f};

        glfwPollEvents();
        poll_input_events();

        m_delta_time = static_cast<float>(now - last_frame_time);

        if (m_delta_time >= m_min_frame_time) {
            m_prev_hovered_widget = m_cur_hovered_widget;
            m_cur_hovered_widget = "";
            m_cur_focused_widget = "";

            if (!m_imgui_vulkan_initialized) {
                std::cerr << "ImGui Vulkan backend not initialized" << std::endl;
                std::exit(-1);
            }

            // Start the Dear ImGui frame
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (!s_once) {
                on_begin_first_loop();
                s_once = true;
            }

            // User Rendering
            loop(m_delta_time);

            // Process debug drawing events
            get_debug_drawer().loop(*this, m_delta_time);

            ImGui::Render();
            render_frame();
            last_frame_time = now;

            // process hover change events
            if (m_prev_hovered_widget != m_cur_hovered_widget) {
                if (m_prev_hovered_widget != k_no_hovered_widget) {
                    // call on_leave_region on the previous widget
                    auto it = m_imgui_window_info.find(m_prev_hovered_widget);
                    if (it != m_imgui_window_info.end()) {
                        it->second.on_leave_region();
                    }
                }

                // call on_enter_region on the current widget
                auto it = m_imgui_window_info.find(m_cur_hovered_widget);
                if (it != m_imgui_window_info.end()) {
                    it->second.on_enter_region();
                }
            }
        }
    }
}

auto GLFWApplication::on_begin_first_loop() -> void {
}

auto GLFWApplication::poll_input_events() noexcept -> void {
    auto screen_dim = glm::ivec2{get_window_width(), get_window_height()};
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    if (!m_last_mouse_pos) {
        m_last_mouse_pos = m_mouse_pos = {x, y};
    } else {
        m_last_mouse_pos = m_mouse_pos;
        m_mouse_pos = {x, y};
    }

    // key events
    for (int i = 0; i < m_key_states.size(); ++i) {
        std::optional<Input> input;
        auto key_state = ImGui::IsKeyDown(static_cast<ImGuiKey>(i));
        if (key_state) {
            if (m_key_states[i]) {
                input = Input{InputType::KEYBOARD, ActionType::HOLD, i};
            } else {
                input = Input{InputType::KEYBOARD, ActionType::PRESS, i};
                m_key_initiated_window[i] = m_cur_hovered_widget;
            }
        } else {
            if (m_key_states[i]) {
                input = Input{InputType::KEYBOARD, ActionType::RELEASE, i};
            }
        }
        if (input) {
            auto event = InputEvent{*input,     m_mouse_pos,          *m_last_mouse_pos,
                                    screen_dim, m_mouse_scroll_delta, m_cur_hovered_widget,
                                    get_time()};
            handle_input(event);
            if (input->action_type == ActionType::RELEASE) {
                m_key_initiated_window[i] = k_no_hovered_widget;
            }
        }
        m_key_states[i] = key_state;
    }

    // mouse events

    // scroll
    if (glm::length(m_mouse_scroll_delta) > 0) {
        auto input = Input{InputType::MOUSE, ActionType::SCROLL, GLFW_MOUSE_BUTTON_MIDDLE};
        handle_input(InputEvent{input, m_mouse_pos, screen_dim, m_mouse_scroll_delta,
                                m_mouse_initiated_window[ImGuiMouseButton_Middle], get_time()});
    }

    for (int i = 0; i < m_mouse_states.size(); ++i) {
        std::optional<Input> input;
        auto mouse_state = ImGui::IsMouseDown(i);
        if (mouse_state) {
            if (m_mouse_states[i]) {
                input = Input{InputType::MOUSE, ActionType::HOLD, i};
            } else {
                input = Input{InputType::MOUSE, ActionType::PRESS, i};
                m_mouse_initiated_window[i] = m_cur_hovered_widget;
            }
        } else {
            if (m_mouse_states[i]) {
                input = Input{InputType::MOUSE, ActionType::RELEASE, i};
            }
        }

        if (input) {
            auto event = InputEvent{*input,     m_mouse_pos,          *m_last_mouse_pos,
                                    screen_dim, m_mouse_scroll_delta, m_mouse_initiated_window[i],
                                    get_time()};
            handle_input(event);
            if (input->action_type == ActionType::RELEASE) {
                m_mouse_initiated_window[i] = k_no_hovered_widget;
            }
        }
        m_mouse_states[i] = mouse_state;
    }
}

auto GLFWApplication::get_window_height() const noexcept -> int {
    int display_h;
    glfwGetFramebufferSize(m_window, nullptr, &display_h);
    return display_h;
}

auto GLFWApplication::get_window_width() const noexcept -> int {
    int display_w;
    glfwGetFramebufferSize(m_window, &display_w, nullptr);
    return display_w;
}

auto GLFWApplication::begin_imgui_window(std::string_view name, ImGuiWindowFlags flags) noexcept
    -> bool {
    auto const ret = ImGui::Begin(name.data(), nullptr, flags);
    if (ImGui::IsWindowHovered(ImGuiItemStatusFlags_HoveredRect)) {
        m_cur_hovered_widget = name;
    }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
        m_cur_focused_widget = name;
    }
    return ret;
}

void GLFWApplication::end_imgui_window() noexcept {
    ImGui::End();
}

auto GLFWApplication::get_window_content_pos(std::string_view name) const noexcept
    -> std::optional<ImVec2> {
    auto const win = ImGui::FindWindowByName(name.data());
    if (!win) {
        return std::nullopt;
    }
    return win->ContentRegionRect.Min;
}

float GLFWApplication::get_time() const noexcept {
    return static_cast<float>(glfwGetTime());
}

float GLFWApplication::get_delta_time() const noexcept {
    return m_delta_time;
}

auto GLFWApplication::init_imgui_vulkan(vk::PhysicalDevice physical_device, vk::Device device,
                                        uint32_t graphics_queue_family, vk::Queue graphics_queue)
    -> void {
    m_vk_physical_device = physical_device;
    m_vk_device = device;
    m_vk_graphics_queue_family = graphics_queue_family;
    m_vk_graphics_queue = graphics_queue;

    create_command_pool();
    create_swapchain();
    create_render_pass();
    create_framebuffers();
    create_command_buffers();
    create_sync_objects();

    auto const pool_sizes = std::array{
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
        vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000},
    };

    m_imgui_descriptor_pool = m_vk_device.createDescriptorPoolUnique(
        vk::DescriptorPoolCreateInfo{}
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
            .setMaxSets(1000 * static_cast<uint32_t>(pool_sizes.size()))
            .setPoolSizes(pool_sizes));

    ImGui_ImplGlfw_InitForVulkan(m_window, true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = m_vk_instance.get();
    init_info.PhysicalDevice = m_vk_physical_device;
    init_info.Device = m_vk_device;
    init_info.QueueFamily = m_vk_graphics_queue_family;
    init_info.Queue = m_vk_graphics_queue;
    init_info.DescriptorPool = m_imgui_descriptor_pool.get();
    init_info.MinImageCount = static_cast<uint32_t>(m_vk_swapchain_images.size());
    init_info.ImageCount = static_cast<uint32_t>(m_vk_swapchain_images.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = [](VkResult err) {
        if (err != VK_SUCCESS) {
            std::cerr << "ImGui Vulkan error: " << err << std::endl;
        }
    };

    ImGui_ImplVulkan_Init(&init_info, m_vk_render_pass.get());

    // Upload fonts
    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        std::cerr << "Failed to upload ImGui Vulkan fonts" << std::endl;
        std::exit(-1);
    }

    m_imgui_vulkan_initialized = true;
}

auto GLFWApplication::shutdown_imgui_vulkan() noexcept -> void {
    if (!m_imgui_vulkan_initialized) {
        return;
    }
    m_vk_device.waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    cleanup_swapchain();
    m_vk_command_pool.reset();
    m_imgui_descriptor_pool.reset();
    m_vk_device = VK_NULL_HANDLE;
    m_imgui_vulkan_initialized = false;
}

auto GLFWApplication::create_vulkan_instance() -> void {
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    uint32_t glfw_ext_count = 0;
    auto glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    auto extensions = std::vector<char const*>{glfw_exts, glfw_exts + glfw_ext_count};
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    auto instance_flags = vk::InstanceCreateFlags{};
#ifdef VK_KHR_portability_enumeration
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instance_flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    auto layers = std::vector<char const*>{};
#ifndef NDEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    auto const app_info = vk::ApplicationInfo{"PTS Editor", VK_MAKE_VERSION(1, 0, 0), "PTS",
                                              VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_2};

    m_vk_instance = vk::createInstanceUnique(
        vk::InstanceCreateInfo{instance_flags, &app_info, layers, extensions});
    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_vk_instance.get());
}

auto GLFWApplication::create_vulkan_surface() -> void {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(m_vk_instance.get(), m_window, nullptr, &surface) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan surface" << std::endl;
        std::exit(-1);
    }
    m_vk_surface = surface;
}

auto GLFWApplication::create_swapchain() -> void {
    auto const caps = m_vk_physical_device.getSurfaceCapabilitiesKHR(m_vk_surface);
    auto const formats = m_vk_physical_device.getSurfaceFormatsKHR(m_vk_surface);
    auto const present_modes = m_vk_physical_device.getSurfacePresentModesKHR(m_vk_surface);

    auto surface_format = formats.front();
    for (auto const& fmt : formats) {
        if (fmt.format == vk::Format::eB8G8R8A8Srgb &&
            fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            surface_format = fmt;
            break;
        }
    }

    auto present_mode = vk::PresentModeKHR::eFifo;
    for (auto const& mode : present_modes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            present_mode = mode;
            break;
        }
    }

    auto extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        extent.width = std::clamp(static_cast<uint32_t>(width), caps.minImageExtent.width,
                                  caps.maxImageExtent.width);
        extent.height = std::clamp(static_cast<uint32_t>(height), caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    m_vk_swapchain_format = surface_format.format;
    m_vk_swapchain_extent = extent;

    auto swapchain_info = vk::SwapchainCreateInfoKHR{}
                              .setSurface(m_vk_surface)
                              .setMinImageCount(image_count)
                              .setImageFormat(m_vk_swapchain_format)
                              .setImageColorSpace(surface_format.colorSpace)
                              .setImageExtent(m_vk_swapchain_extent)
                              .setImageArrayLayers(1)
                              .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                              .setImageSharingMode(vk::SharingMode::eExclusive)
                              .setPreTransform(caps.currentTransform)
                              .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                              .setPresentMode(present_mode)
                              .setClipped(true);

    m_vk_swapchain = m_vk_device.createSwapchainKHRUnique(swapchain_info);
    m_vk_swapchain_images = m_vk_device.getSwapchainImagesKHR(m_vk_swapchain.get());

    m_vk_swapchain_image_views.clear();
    m_vk_swapchain_image_views.reserve(m_vk_swapchain_images.size());
    for (auto const& image : m_vk_swapchain_images) {
        auto view = m_vk_device.createImageViewUnique(
            vk::ImageViewCreateInfo{}
                .setImage(image)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(m_vk_swapchain_format)
                .setSubresourceRange(vk::ImageSubresourceRange{}
                                         .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                         .setBaseMipLevel(0)
                                         .setLevelCount(1)
                                         .setBaseArrayLayer(0)
                                         .setLayerCount(1)));
        m_vk_swapchain_image_views.emplace_back(std::move(view));
    }
}

auto GLFWApplication::create_render_pass() -> void {
    auto color_attachment = vk::AttachmentDescription{}
                                .setFormat(m_vk_swapchain_format)
                                .setSamples(vk::SampleCountFlagBits::e1)
                                .setLoadOp(vk::AttachmentLoadOp::eClear)
                                .setStoreOp(vk::AttachmentStoreOp::eStore)
                                .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                .setInitialLayout(vk::ImageLayout::eUndefined)
                                .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto color_ref = vk::AttachmentReference{}.setAttachment(0).setLayout(
        vk::ImageLayout::eColorAttachmentOptimal);

    auto subpass = vk::SubpassDescription{}
                       .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                       .setColorAttachments(color_ref);

    auto dependency = vk::SubpassDependency{}
                          .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                          .setDstSubpass(0)
                          .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                          .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                          .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

    m_vk_render_pass = m_vk_device.createRenderPassUnique(vk::RenderPassCreateInfo{}
                                                              .setAttachments(color_attachment)
                                                              .setSubpasses(subpass)
                                                              .setDependencies(dependency));
}

auto GLFWApplication::create_framebuffers() -> void {
    m_vk_framebuffers.clear();
    m_vk_framebuffers.reserve(m_vk_swapchain_image_views.size());
    for (auto const& view : m_vk_swapchain_image_views) {
        auto framebuffer =
            m_vk_device.createFramebufferUnique(vk::FramebufferCreateInfo{}
                                                    .setRenderPass(m_vk_render_pass.get())
                                                    .setAttachments(view.get())
                                                    .setWidth(m_vk_swapchain_extent.width)
                                                    .setHeight(m_vk_swapchain_extent.height)
                                                    .setLayers(1));
        m_vk_framebuffers.emplace_back(std::move(framebuffer));
    }
}

auto GLFWApplication::create_command_pool() -> void {
    m_vk_command_pool = m_vk_device.createCommandPoolUnique(
        vk::CommandPoolCreateInfo{}
            .setQueueFamilyIndex(m_vk_graphics_queue_family)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer));
}

auto GLFWApplication::create_command_buffers() -> void {
    static constexpr uint32_t k_frames_in_flight = 2;
    m_vk_command_buffers = m_vk_device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{
        m_vk_command_pool.get(), vk::CommandBufferLevel::ePrimary, k_frames_in_flight});
}

auto GLFWApplication::create_sync_objects() -> void {
    static constexpr uint32_t k_frames_in_flight = 2;
    m_vk_image_available_semaphores.resize(k_frames_in_flight);
    m_vk_render_finished_semaphores.resize(k_frames_in_flight);
    m_vk_in_flight_fences.resize(k_frames_in_flight);
    for (uint32_t i = 0; i < k_frames_in_flight; ++i) {
        m_vk_image_available_semaphores[i] = m_vk_device.createSemaphoreUnique({});
        m_vk_render_finished_semaphores[i] = m_vk_device.createSemaphoreUnique({});
        m_vk_in_flight_fences[i] = m_vk_device.createFenceUnique(
            vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled));
    }
}

auto GLFWApplication::cleanup_swapchain() -> void {
    if (!m_vk_device) {
        return;
    }
    m_vk_device.waitIdle();
    m_vk_framebuffers.clear();
    m_vk_swapchain_image_views.clear();
    m_vk_render_pass.reset();
    m_vk_swapchain.reset();
}

auto GLFWApplication::recreate_swapchain() -> void {
    int width = 0;
    int height = 0;
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(m_window, &width, &height);
        glfwWaitEvents();
    }
    cleanup_swapchain();
    create_swapchain();
    create_render_pass();
    create_framebuffers();
    ImGui_ImplVulkan_SetMinImageCount(static_cast<uint32_t>(m_vk_swapchain_images.size()));
}

auto GLFWApplication::record_command_buffer(vk::CommandBuffer cmd_buf, uint32_t image_index)
    -> void {
    cmd_buf.begin(vk::CommandBufferBeginInfo{});
    auto clear_value = vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0, 0, 0, 1}}};
    auto render_pass_info = vk::RenderPassBeginInfo{}
                                .setRenderPass(m_vk_render_pass.get())
                                .setFramebuffer(m_vk_framebuffers[image_index].get())
                                .setRenderArea(vk::Rect2D{{0, 0}, m_vk_swapchain_extent})
                                .setClearValues(clear_value);

    cmd_buf.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd_buf);
    cmd_buf.endRenderPass();
    cmd_buf.end();
}

auto GLFWApplication::render_frame() -> void {
    if (!m_imgui_vulkan_initialized) {
        return;
    }

    static constexpr uint32_t k_frames_in_flight = 2;
    auto const frame_index = m_vk_frame_index % k_frames_in_flight;
    (void) m_vk_device.waitForFences(m_vk_in_flight_fences[frame_index].get(), true, UINT64_MAX);

    uint32_t image_index = 0;
    auto acquire_result = m_vk_device.acquireNextImageKHR(
        m_vk_swapchain.get(), UINT64_MAX, m_vk_image_available_semaphores[frame_index].get(), {},
        &image_index);
    if (acquire_result == vk::Result::eErrorOutOfDateKHR ||
        acquire_result == vk::Result::eSuboptimalKHR) {
        recreate_swapchain();
        return;
    }

    m_vk_device.resetFences(m_vk_in_flight_fences[frame_index].get());
    m_vk_command_buffers[frame_index]->reset();
    record_command_buffer(m_vk_command_buffers[frame_index].get(), image_index);

    auto const wait_stages =
        std::array{vk::PipelineStageFlags{vk::PipelineStageFlagBits::eColorAttachmentOutput}};
    auto submit_info = vk::SubmitInfo{}
                           .setWaitSemaphores(m_vk_image_available_semaphores[frame_index].get())
                           .setWaitDstStageMask(wait_stages)
                           .setCommandBuffers(m_vk_command_buffers[frame_index].get())
                           .setSignalSemaphores(m_vk_render_finished_semaphores[frame_index].get());

    m_vk_graphics_queue.submit(submit_info, m_vk_in_flight_fences[frame_index].get());

    auto present_info = vk::PresentInfoKHR{}
                            .setWaitSemaphores(m_vk_render_finished_semaphores[frame_index].get())
                            .setSwapchains(m_vk_swapchain.get())
                            .setImageIndices(image_index);
    auto present_result = m_vk_graphics_queue.presentKHR(present_info);

    if (present_result == vk::Result::eErrorOutOfDateKHR ||
        present_result == vk::Result::eSuboptimalKHR || m_vk_framebuffer_resized) {
        m_vk_framebuffer_resized = false;
        recreate_swapchain();
    }

    m_vk_frame_index = (m_vk_frame_index + 1) % k_frames_in_flight;
}
