#include "include/glfwApplication.h"

#include <imgui_internal.h>

#include "imgui/imhelper.h"
#include <iostream>

using namespace PTS;

// stubs for callbacks
namespace PTS {
    static void click_func(GLFWwindow* window, int button, int action, int mods) {
        // auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
        // check if ImGui is using the mouse
        //static_cast<void>(mods);
        //app->m_mouse_states[button] = action == GLFW_PRESS;
    }
    static void motion_func(GLFWwindow* window, double x, double y) {

    }
    static void scroll_func(GLFWwindow* window, double x, double y) {
        auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
        app->m_mouse_scroll_delta = { x, y };
    }
    static void key_func(GLFWwindow* window, int key, int scancode, int action, int mods) {
        //auto const app = static_cast<GLFWApplication*>(glfwGetWindowUserPointer(window));
        //static_cast<void>(scancode);
        //static_cast<void>(mods);
        //app->m_key_states[key] = action == GLFW_PRESS;
    }
    static void error_func(int error, const char* description) {
        std::cerr << "GLFW error: " << error << ": " << description << std::endl;
        std::exit(-1);
    }
}

GLFWApplication::GLFWApplication(std::string_view name, unsigned width, unsigned height, float min_frame_time) {
    set_min_frame_time(min_frame_time);
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
        m_mouse_scroll_delta = glm::vec2{ 0.0f };

        glfwPollEvents();
        poll_input_events();

        m_delta_time = static_cast<float>(now - last_frame_time);
        m_log_flush_timer += m_delta_time;

        if (m_delta_time >= m_min_frame_time) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

            m_prev_hovered_widget = m_cur_hovered_widget;
            m_cur_hovered_widget = "";
            m_cur_focused_widget = "";

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

        if (m_log_flush_timer >= m_log_flush_interval) {
            m_log_flush_timer = 0.0f;
            clear_logs();
        }
    }
}

auto GLFWApplication::on_begin_first_loop() -> void {}

auto GLFWApplication::poll_input_events() noexcept -> void {
    auto screen_dim = glm::ivec2{ get_window_width(), get_window_height() };
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    if(!m_last_mouse_pos) {
        m_last_mouse_pos = m_mouse_pos = { x, y };
    } else {
        m_last_mouse_pos = m_mouse_pos;
        m_mouse_pos = { x, y };
    }

    // key events
    for(int i=0; i<m_key_states.size(); ++i) {
        std::optional<Input> input;
        auto key_state = ImGui::IsKeyDown(static_cast<ImGuiKey>(i));
        if (key_state) {
            if (m_key_states[i]) {
                input = Input {
                    InputType::KEYBOARD,
                    ActionType::HOLD,
                    i
                };
            } else {
                input = Input {
                    InputType::KEYBOARD,
                    ActionType::PRESS,
                    i
                };
                m_key_initiated_window[i] = m_cur_hovered_widget;
            }
        } else {
            if (m_key_states[i]) {
                input = Input {
                    InputType::KEYBOARD,
                    ActionType::RELEASE,
                    i
                };
            }
        }
        if(input) {
            auto event = InputEvent {
                *input,
                m_mouse_pos,
                *m_last_mouse_pos,
                screen_dim,
                m_mouse_scroll_delta,
                m_cur_hovered_widget,
                get_time()
            };
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
        auto input = Input{
            InputType::MOUSE, ActionType::SCROLL, GLFW_MOUSE_BUTTON_MIDDLE
        };
        handle_input(InputEvent{
            input,
            m_mouse_pos,
            screen_dim,
            m_mouse_scroll_delta,
            m_mouse_initiated_window[ImGuiMouseButton_Middle],
            get_time()
        });
    }

    for(int i=0; i<m_mouse_states.size(); ++i) {
        std::optional<Input> input;
        auto mouse_state = ImGui::IsMouseDown(i);
        if (mouse_state) {
            if (m_mouse_states[i]) {
                input = Input {
                    InputType::MOUSE,
                    ActionType::HOLD,
                    i
                };
            } else {
                input = Input {
                    InputType::MOUSE,
                    ActionType::PRESS,
                    i
                };
                m_mouse_initiated_window[i] = m_cur_hovered_widget;
            }
        } else {
	        if (m_mouse_states[i]) {
		        input = Input{
			        InputType::MOUSE,
			        ActionType::RELEASE,
			        i
		        };
	        }
        }

        if (input) {
            auto event = InputEvent {
                *input,
                m_mouse_pos,
                *m_last_mouse_pos,
                screen_dim,
                m_mouse_scroll_delta,
                m_mouse_initiated_window[i],
                get_time()
            };
            handle_input(event);
            if (input->action_type == ActionType::RELEASE) {
                m_mouse_initiated_window[i] = k_no_hovered_widget;
            }
        }
        m_mouse_states[i] = mouse_state;
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
    ImGuiWindowFlags flags
) noexcept -> bool {
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
