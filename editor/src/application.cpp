#include "include/application.h"
#include "include/imgui/imhelper.h"

#include <imgui_internal.h>

// stubs for callbacks
static void click_func(GLFWwindow* window, int button, int action, int mods) {
    auto const app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    // check if ImGui is using the mouse
    if (ImGui::GetIO().WantCaptureMouse && !app->mouse_over_any_event_region()) {
        return;
    }
    app->mouse_clicked(button, action, mods);
}
static void motion_func(GLFWwindow* window, double x, double y) {
    auto const app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse && !app->mouse_over_any_event_region()) {
        return;
    }
    app->cursor_moved(x, y);
}
static void scroll_func(GLFWwindow* window, double x, double y) {
    auto const app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse && !app->mouse_over_any_event_region()) {
        return;
    }
    app->mouse_scroll(x, y);
}
static void key_func(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto const app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureKeyboard && !app->mouse_over_any_event_region()) {
        return;
    }
    app->key_pressed(key, scancode, action, mods);
}

static void error_func(int error, const char* description) {
    std::cerr << "GLFW error: " << error << ": " << description << std::endl;
    Application::quit(-1);
}

Application::Application(Renderer& renderer, Scene& scene, std::string_view name)
    : m_scene { scene }, m_renderer{ renderer },
      m_cam{ renderer.get_config().fovy, renderer.get_config().width, renderer.get_config().height, scene.get_good_cam_start() }
{
    if (s_app) {
        std::cerr << "There can only be one instance of application" << std::endl;
        Application::quit(-1);
    } else {
        s_app = this;
    }

    glfwSetErrorCallback(error_func);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        quit(-1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(renderer.get_config().width, renderer.get_config().height, name.data(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create window" << std::endl;
        quit(-1);
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
        quit(-1);
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    ImGui::SetNextWindowPos({ 10, 10 });
    ImGui::SetNextWindowSize({ 0, static_cast<float>(renderer.get_config().height) / 5.0f });

    // config camera and initialize renderer
    check_error(get_renderer().open_scene(scene));
}

Application::~Application() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
}

void Application::run() {
    double last_frame_time = 0;
    while (!glfwWindowShouldClose(m_window)) {
        double now = glfwGetTime();

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        double dt = now - last_frame_time;
        if (dt >= m_renderer.get_config().min_frame_time) {
            m_prev_hovered_widget = m_cur_hovered_widget;

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // User Rendering
            loop(dt);

            // Process debug drawing events
            get_debug_drawer().loop(dt);

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
                    auto it = m_imgui_window_info.find(m_prev_hovered_widget);
                    if (it != m_imgui_window_info.end() && it->second.on_leave_region) {
                        it->second.on_leave_region.value()();
                    }
                }
            }
        }

        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
    }
}

auto Application::get_window_height() const noexcept->int {
    int display_h;
    glfwGetFramebufferSize(m_window, nullptr, &display_h);
    return display_h;
}

auto Application::get_window_width() const noexcept->int {
    int display_w;
    glfwGetFramebufferSize(m_window, &display_w, nullptr);
    return display_w;
}

void Application::begin_imgui_window(
    std::string_view name, 
    bool recv_mouse_event,
    ImGuiWindowFlags flags,
    std::optional<std::function<void()>> const& on_leave_region
) noexcept {
    // NOTE: here we assume that recv_mouse_event will not change
    // in the lifetime of the application
    // i.e., the following code pattern will not happen:
    // if (cond) begin_imgui_window("window1", true);
    // else begin_imgui_window("window1", false);

    if (!m_imgui_window_info.count(name)) {
        m_imgui_window_info[name] = ImGuiWindowInfo {
            recv_mouse_event,
            on_leave_region
        };
    }

    ImGui::Begin(name.data(), nullptr, flags);
    if (ImGui::IsWindowHovered()) {
        m_cur_hovered_widget = name;
    }

    // disable alt key for imgui
    ImGui::SetKeyOwner(ImGuiMod_Alt, ImGui::GetCurrentWindow()->ID);
}

void Application::end_imgui_window() noexcept {
    ImGui::End();
}

auto Application::get_window_content_pos(std::string_view name) const noexcept -> std::optional<ImVec2> {
    auto win = ImGui::FindWindowByName(name.data());
    if (!win) {
        return std::nullopt;
    }
    return win->ContentRegionRect.Min;
}

bool Application::mouse_over_any_event_region() const noexcept {
    auto it = m_imgui_window_info.find(m_cur_hovered_widget);
    auto ret = it != m_imgui_window_info.end() && it->second.can_recv_mouse_event;
    return ret;
}

void Application::quit(int code) {
    if (s_app) {
        s_app->~Application();
    }
	exit(code);
}

