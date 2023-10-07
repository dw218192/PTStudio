#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "scene.h"
#include "renderer.h"
#include "ext.h"

/**
 * \brief simple wrapper to make it easier to create an application\n
 * There can only be one application at a time. Be sure not to create multiple applications.
*/
struct Application {
    Application(RenderConfig const& config, std::string_view name);
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
	virtual void cursor_moved([[maybe_unused]] double x, [[maybe_unused]] double y) { }

    /**
     * \brief Called when the mouse is clicked. Override to handle mouse clicks.
     * \param button the button that was clicked
     * \param action the action that was performed
     * \param mods the modifiers that were pressed
     * \see https://www.glfw.org/docs/latest/input_guide.html
    */
    virtual void mouse_clicked([[maybe_unused]] int button, [[maybe_unused]] int action, [[maybe_unused]] int mods) { }
    
    /**
     * \brief Called when the user scrolls. Override to handle mouse scrolling.
     * \param x the x offset of the scroll
     * \param y the y offset of the scroll
     * \see https://www.glfw.org/docs/latest/input_guide.html
    */
    virtual void mouse_scroll([[maybe_unused]] double x, [[maybe_unused]] double y) { }

    /**
     * \brief Called every frame. This function has to be implemented.\n
     * Typically this involves updating the scene and rendering it by calling Renderer::render().\n
     * You may also render imgui here.
     * \see Renderer::render()
    */
    virtual void loop() = 0;

    /**
     * \brief Checks a result returned from some function. Prints the error and Terminates the program on error.
     * \tparam T Type of the real return value
     * \tparam E Type of the error return value
     * \param res the result
     * \return The real return value if no error
     */
    template<typename T, typename E>
    static constexpr T check_error(Result<T, E> const& res);

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
        assert(s_app);
    	return *s_app;
    }

protected:
    /**
     * \brief Gets the renderer for the application.
     * \return the renderer
    */
    [[nodiscard]] auto get_renderer() -> Renderer& { return m_renderer; }
private:
    static inline Application* s_app = nullptr;
    GLFWwindow* m_window;
    Renderer m_renderer;
};

template <typename T, typename E>
constexpr T Application::check_error(Result<T, E> const& res) {
    if(!res.valid()) {
        std::cerr << res.error() << std::endl;
        Application::quit(-1);
    }
    return res.value();
}
