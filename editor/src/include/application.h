#pragma once

#include "scene.h"
#include "renderer.h"

#include <unordered_set>
#include <iostream>
#include <GLFW/glfw3.h>
#include <functional>
#include <optional>

#include "debugDrawer.h"

/**
 * \brief simple wrapper to make it easier to create an application\n
 * There can only be one application at a time. Be sure not to create multiple applications.
*/
struct Application {
    Application(Renderer& renderer, Scene& scene, std::string_view name);
    Application(Application const&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application const&) = delete;
    Application& operator=(Application&&) = delete;
    virtual ~Application();

    /**
     * \brief Runs the main loop of the application.
    */
	void run();

    [[nodiscard]] auto get_window_width() const noexcept -> int;
    [[nodiscard]] auto get_window_height() const noexcept -> int;
    
    /**
     * \brief Called when the cursor moves. Override to handle cursor movement.
     * \param x the x position of the cursor
     * \param y the y position of the cursor
     * \see https://www.glfw.org/docs/latest/input_guide.html
    */
	virtual void cursor_moved(double x, double y) { }

    /**
     * \brief Called when the mouse is clicked. Override to handle mouse clicks.
     * \param button the button that was clicked
     * \param action the action that was performed
     * \param mods the modifiers that were pressed
     * \see https://www.glfw.org/docs/latest/input_guide.html
    */
    virtual void mouse_clicked(int button, int action, int mods) { }
    
    /**
     * \brief Called when the user scrolls. Override to handle mouse scrolling.
     * \param x the x offset of the scroll
     * \param y the y offset of the scroll
     * \see https://www.glfw.org/docs/latest/input_guide.html
    */
    virtual void mouse_scroll(double x, double y) { }

    /**
     * \brief Called when a key is pressed. Override to handle key presses.
     * \param key the key that was pressed
     * \param scancode the scancode of the key
     * \param action the action that was performed
     * \param mods the modifiers that were pressed
     * \see https://www.glfw.org/docs/latest/input_guide.html
    */
    virtual void key_pressed(int key, int scancode, int action, int mods) { }

    /**
     * \brief Called every frame. Override to handle the main loop.
     * \param dt the time since the last frame
    */
    virtual void loop(float dt) = 0;

    /**
     * \brief Sets a new scene for the application
     * \param new_scene the new scene to use
     */
    static void set_scene(Scene const& new_scene) {
	    get_application().m_scene = new_scene;
    }

    [[nodiscard]] static auto get_scene() -> Scene& {
	    return get_application().m_scene;
    }
    [[nodiscard]] static auto get_cam() -> Camera& {
	    return get_application().m_cam;
    }

    /**
     * \brief Checks a result returned from some function. Prints the error and Terminates the program on error.
     * \tparam T Type of the real return value
     * \tparam E Type of the error return value
     * \param res the result
     * \return The real return value if no error
     */
    template<typename T, typename E>
    static constexpr decltype(auto) check_error(tl::expected<T, E> const& res);
    template<typename T, typename E>
    static constexpr decltype(auto) check_error(tl::expected<T, E>&& res);
    /**
     * \brief Terminates the program with the given exit code
     * \param code the exit code
    */
	[[noreturn]] static void quit(int code);

    /**
     * \brief Returns the application instance
     * \return the application instance
     * \note This function is only valid after the application has been created
    */
    [[nodiscard]] static auto get_application() noexcept -> Application& {
    	return *s_app;
    }
    [[nodiscard]] bool mouse_over_any_event_region() const noexcept;
protected:   
    /**
     * \brief Gets the renderer for the application.
     * \return the renderer
    */
    [[nodiscard]] auto get_renderer() const -> Renderer& { return m_renderer; }
    [[nodiscard]] auto get_debug_drawer() -> DebugDrawer& { return m_debug_drawer; }
    
    [[nodiscard]] auto get_cur_hovered_widget() const noexcept -> std::string_view { return m_cur_hovered_widget; }
    

    // imgui helpers
    void begin_imgui_window(
        std::string_view name, 
        bool recv_mouse_event = false,
        ImGuiWindowFlags flags = 0,
        std::optional<std::function<void()>> const& on_leave_region = std::nullopt
    ) noexcept;

    void end_imgui_window() noexcept;
    auto get_window_pos(std::string_view name) const noexcept -> std::optional<ImVec2>;

private:
    static inline Application* s_app = nullptr;
    GLFWwindow* m_window;
    Scene& m_scene;
    Renderer& m_renderer;
    Camera m_cam;
    DebugDrawer m_debug_drawer;

    // these are used to check if the mouse is over any mouse event region
    struct ImGuiWindowInfo {
        bool can_recv_mouse_event;
        std::optional<std::function<void()>> on_leave_region;
    };
    std::unordered_map<std::string_view, ImGuiWindowInfo> m_imgui_window_info;
    std::string_view m_cur_hovered_widget, m_prev_hovered_widget;
    static constexpr auto k_no_hovered_widget = "";
};

template <typename T, typename E>
constexpr decltype(auto) Application::check_error(tl::expected<T, E> const& res) {
    if(!res) {
        std::cerr << res.error() << std::endl;
        Application::quit(-1);
    }
    if constexpr (!std::is_void_v<T>) {
        return res.value();
    }
}
template <typename T, typename E>
constexpr decltype(auto) Application::check_error(tl::expected<T, E>&& res) {
    if (!res) {
        std::cerr << res.error() << std::endl;
        Application::quit(-1);
    }
	if constexpr (!std::is_void_v<T>) {
        return std::move(res).value();
    }
}