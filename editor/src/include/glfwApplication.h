#pragma once

#include "ext.h"

#include <functional>
#include <optional>

#include "debugDrawer.h"
#include "application.h"

struct GLFWApplication : Application {
    NO_COPY_MOVE(GLFWApplication);

    GLFWApplication(std::string_view name, unsigned width, unsigned height, float min_frame_time);
    ~GLFWApplication() override;

	void run() override;

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

    [[nodiscard]] bool mouse_over_any_event_region() const noexcept;
protected:
    virtual void on_begin_first_loop() { }

    /**
     * \brief Gets the renderer for the application.
     * \return the renderer
    */
    [[nodiscard]] auto get_debug_drawer() -> DebugDrawer& { return m_debug_drawer; }
    [[nodiscard]] auto get_cur_hovered_widget() const noexcept { return m_cur_hovered_widget; }
    [[nodiscard]] auto get_cur_focused_widget() const noexcept { return m_cur_focused_widget; }
    
    // imgui helpers
    void begin_imgui_window(
        std::string_view name, 
        ImGuiWindowFlags flags = 0,
        std::optional<std::function<void()>> const& on_leave_region = std::nullopt,
        std::optional<std::function<void()>> const& on_enter_region = std::nullopt
    ) noexcept;

    void end_imgui_window() noexcept;
    auto get_window_content_pos(std::string_view name) const noexcept -> std::optional<ImVec2>;

private:
    GLFWwindow* m_window;
    DebugDrawer m_debug_drawer;
    float m_min_frame_time;

    // used to help detect if the mouse enters/leaves certain imgui windows
    struct ImGuiWindowInfo {
        std::optional<std::function<void()>> on_leave_region, on_enter_region;
    };
    std::unordered_map<std::string_view, ImGuiWindowInfo> m_imgui_window_info;
    
    std::string_view m_cur_hovered_widget, m_prev_hovered_widget;
    std::string_view m_cur_focused_widget;

    static constexpr auto k_no_hovered_widget = "";
};
