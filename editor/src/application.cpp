#include "include/application.h"

// stubs for callbacks
static void clickFunc(GLFWwindow* window, int button, int action, int mods) {
    auto const app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    // check if ImGui is using the mouse
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    app->mouse_clicked(button, action, mods);
}
static void motionFunc(GLFWwindow* window, double x, double y) {
    auto const app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    app->cursor_moved(x, y);
}
static void scrollFunc(GLFWwindow* window, double x, double y) {
    auto const app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }
    app->mouse_scroll(x, y);
}
static void errorFunc(int error, const char* description) {
    std::cerr << "GLFW error: " << error << ": " << description << std::endl;
    Application::quit(-1);
}

Application::Application(Renderer& renderer, Scene& scene, std::string_view name)
    : m_scene { scene }, m_renderer{ renderer },
      m_cam{ renderer.get_config().fovy, renderer.get_config().width, renderer.get_config().height, Transform {} }
{
    if (s_app) {
        std::cerr << "There can only be one instance of application" << std::endl;
        Application::quit(-1);
    } else {
        s_app = this;
    }

    glfwSetErrorCallback(errorFunc);

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
    glfwSetMouseButtonCallback(m_window, clickFunc);
    glfwSetCursorPosCallback(m_window, motionFunc);
    glfwSetScrollCallback(m_window, scrollFunc);

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

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    ImGui::SetNextWindowPos({ 10, 10 });
    ImGui::SetNextWindowSize({ 0, static_cast<float>(renderer.get_config().height) / 5.0f });
    // ImGui::GetIO().IniFilename = nullptr;

    // config camera and initialize renderer
    get_cam().set_view_transform(scene.get_good_cam_start());
    check_error(get_renderer().open_scene(scene));
}

Application::~Application() {
    glfwTerminate();
}

void Application::run() {
    double last_frame_time = 0;
    while (!glfwWindowShouldClose(m_window)) {
        double now = glfwGetTime();

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        if (now - last_frame_time >= m_renderer.get_config().min_frame_time) {
            // Do Rendering
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // User Rendering
            loop();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(m_window);
            last_frame_time = now;
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

void Application::quit(int code) {
    exit(code);
}
