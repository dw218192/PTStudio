#include "include/glfwApplication.h"

#include <imgui_internal.h>

#include "include/imgui/imhelper.h"
#include <iostream>

// stubs for callbacks
static void click_func(GLFWwindow* window, int button, int action, int mods) {
    auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    // check if ImGui is using the mouse
    if (ImGui::GetIO().WantCaptureMouse && !app->mouse_over_any_event_region()) {
        return;
    }
    app->mouse_clicked(button, action, mods);
}
static void motion_func(GLFWwindow* window, double x, double y) {
    auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse && !app->mouse_over_any_event_region()) {
        return;
    }
    app->cursor_moved(x, y);
}
static void scroll_func(GLFWwindow* window, double x, double y) {
    auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse && !app->mouse_over_any_event_region()) {
        return;
    }
    app->mouse_scroll(x, y);
}
static void key_func(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureKeyboard && !app->mouse_over_any_event_region()) {
        return;
    }
    app->key_pressed(key, scancode, action, mods);
}

static void error_func(int error, const char* description) {
    std::cerr << "GLFW error: " << error << ": " << description << std::endl;
    std::exit(-1);
}

GLFWApplication::GLFWApplication(std::string_view name, unsigned width, unsigned height, float min_frame_time)
    : m_min_frame_time (min_frame_time) 
{
    glfwSetErrorCallback(error_func);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        std::exit(-1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create window" << std::endl;
        std::exit(-1);
    }
    
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);

    // set callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, click_func);
    glfwSetCursorPosCallback(m_window, motion_func);
    glfwSetScrollCallback(m_window, scroll_func);
    glfwSetKeyCallback(m_window, key_func);

    // initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        std::exit(-1);
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    ImGui::SetNextWindowPos({ 10, 10 });
    ImGui::SetNextWindowSize({ 0, static_cast<float>(height) / 5.0f });
}

GLFWApplication::~GLFWApplication() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
}

void GLFWApplication::run() {
    static bool s_once = false;
    double last_frame_time = 0;
    while (!glfwWindowShouldClose(m_window)) {
        auto const now = glfwGetTime();

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        m_delta_time = static_cast<float>(now - last_frame_time);
        if (m_delta_time >= m_min_frame_time) {
            m_prev_hovered_widget = m_cur_hovered_widget;

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
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
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        	ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(m_window);

            glfwSwapBuffers(m_window);
            last_frame_time = now;

            // process hover change events
            if (m_prev_hovered_widget != m_cur_hovered_widget) {
                if (m_prev_hovered_widget != k_no_hovered_widget) {

                    // call on_leave_region on the previous widget
                    auto it = m_imgui_window_info.find(m_prev_hovered_widget);
                    if (it != m_imgui_window_info.end() && it->second.on_leave_region.has_value()) {
                        it->second.on_leave_region.value()();
                    }

                    // call on_enter_region on the current widget
                    it = m_imgui_window_info.find(m_cur_hovered_widget);
                    if (it != m_imgui_window_info.end() && it->second.on_enter_region.has_value()) {
                        it->second.on_enter_region.value()();
                    }
                }
            }
        }
        /*
        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
        */
    }
}

auto GLFWApplication::get_window_height() const noexcept->int {
    int display_h;
    glfwGetFramebufferSize(m_window, nullptr, &display_h);
    return display_h;
}

auto GLFWApplication::get_window_width() const noexcept->int {
    int display_w;
    glfwGetFramebufferSize(m_window, &display_w, nullptr);
    return display_w;
}

auto GLFWApplication::begin_imgui_window(
    std::string_view name, 
    ImGuiWindowFlags flags,
    std::optional<std::function<void()>> const& on_leave_region,
    std::optional<std::function<void()>> const& on_enter_region
) noexcept -> bool {
    if (!m_imgui_window_info.count(name)) {
        m_imgui_window_info[name] = ImGuiWindowInfo {
            on_leave_region,
            on_enter_region
        };
    }

    auto const ret = ImGui::Begin(name.data(), nullptr, flags);
    if (ImGui::IsWindowHovered()) {
        m_cur_hovered_widget = name;
        /*
        if (ImGui::GetMousePos() >= ImGui::GetWindowPos() && ImGui::GetMousePos() <= ImGui::GetWindowPos() + ImGui::GetWindowSize()) {
            m_cur_hovered_widget = name;
        }
        */
    }

    // record the current focus
    if (ImGui::IsWindowFocused()) {
        m_cur_focused_widget = name;
    }
    return ret;
}

void GLFWApplication::end_imgui_window() noexcept {
    ImGui::End();
}

auto GLFWApplication::get_window_content_pos(std::string_view name) const noexcept -> std::optional<ImVec2> {
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

bool GLFWApplication::mouse_over_any_event_region() const noexcept {
    auto it = m_imgui_window_info.find(m_cur_hovered_widget);
    auto ret = it != m_imgui_window_info.end();
    return ret;
}
