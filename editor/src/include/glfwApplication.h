#pragma once

#include "ext.h"

#include <functional>
#include <optional>
#include <string_view>
#include <bitset>
#include <array>

#include "debugDrawer.h"
#include "application.h"
#include "inputAction.h"

namespace PTS {
	/**
	 * \brief abstract GLFW application. Responsible for creating the window and polling events.
	*/
	struct GLFWApplication : Application {
		friend static void click_func(GLFWwindow* window, int button, int action, int mods);
		friend static void motion_func(GLFWwindow* window, double x, double y);
		friend static void scroll_func(GLFWwindow* window, double x, double y);
		friend static void key_func(GLFWwindow* window, int key, int scancode, int action, int mods);
		friend static void error_func(int error, const char* description);
		
		NO_COPY_MOVE(GLFWApplication);

		GLFWApplication(std::string_view name, unsigned width, unsigned height, float min_frame_time);
		~GLFWApplication() override;

		void run() override;

		[[nodiscard]] auto get_window_width() const noexcept -> int;
		[[nodiscard]] auto get_window_height() const noexcept -> int;
		/**
		 * \brief Called every frame. Override to handle the main loop.
		 * \param dt the time since the last frame
		*/
		virtual void loop(float dt) = 0;

	protected:
		virtual auto handle_input(InputEvent const& event) noexcept -> void {}
		virtual auto on_begin_first_loop() -> void {}
		auto poll_input_events() noexcept -> void;

		/**
		 * \brief Gets the renderer for the application.
		 * \return the renderer
		*/
		[[nodiscard]] auto get_debug_drawer() -> DebugDrawer& { return m_debug_drawer; }
		[[nodiscard]] auto get_cur_hovered_widget() const noexcept { return m_cur_hovered_widget; }
		[[nodiscard]] auto get_cur_focused_widget() const noexcept { return m_cur_focused_widget; }

		// imgui helpers
		auto begin_imgui_window(
			std::string_view name,
			ImGuiWindowFlags flags = 0,
			std::optional<std::function<void()>> const& on_leave_region = std::nullopt,
			std::optional<std::function<void()>> const& on_enter_region = std::nullopt
		) noexcept -> bool;

		void end_imgui_window() noexcept;
		auto get_window_content_pos(std::string_view name) const noexcept -> std::optional<ImVec2>;

		[[nodiscard]] auto get_time() const noexcept -> float override;
		[[nodiscard]] auto get_delta_time() const noexcept -> float override;

	private:
		glm::vec2 m_mouse_scroll_delta;
		glm::vec2 m_mouse_pos;
		std::optional<glm::vec2> m_last_mouse_pos { std::nullopt };
		std::bitset<ImGuiMouseButton_COUNT> m_mouse_states {};
		std::bitset<ImGuiKey_COUNT> m_key_states {};
		std::array<std::string_view, ImGuiMouseButton_COUNT> m_mouse_initiated_window {};
		std::array<std::string_view, ImGuiKey_COUNT> m_key_initiated_window {};

		GLFWwindow* m_window;
		DebugDrawer m_debug_drawer;
		float m_min_frame_time;
		float m_delta_time{0.0f};

		// used to help detect if the mouse enters/leaves certain imgui windows
		struct ImGuiWindowInfo {
			std::optional<std::function<void()>> on_leave_region, on_enter_region;
		};

		std::unordered_map<std::string_view, ImGuiWindowInfo> m_imgui_window_info;

		std::string_view m_cur_hovered_widget, m_prev_hovered_widget;
		std::string_view m_cur_focused_widget;

		static constexpr auto k_no_hovered_widget = "";
	};
}
