#pragma once

#include <array>
#include <imgui.h>
#include <ImGuizmo.h>
#include <iostream>

#include "editorRenderer.h"
#include "scene.h"

#include "archive.h"
#include "camera.h"
#include "glfwApplication.h"
#include "glTexture.h"
#include "inputAction.h"
#include "singleton.h"

namespace PTS::Editor {
	constexpr auto k_init_move_sensitivity = 5.0f;
	constexpr auto k_init_rot_sensitivity = 60.0f;
	constexpr auto k_object_select_mouse_time = 1.0f;
	constexpr auto k_default_renderer_idx = 0;

	struct EditorApplication final : GLFWApplication, Singleton<EditorApplication> {
		friend Singleton;
		NO_COPY_MOVE(EditorApplication);

		auto add_renderer(std::unique_ptr<Renderer> renderer) noexcept -> void;
		auto loop(float dt) -> void override;
		auto quit(int code) -> void override;
		/**
			 * \brief Checks a result returned from some function. Prints the error and Terminates the program on error.
			 * \tparam T Type of the real return value
			 * \tparam E Type of the error return value
			 * \param res the result
			 * \return The real return value if no error
			 */
		template <typename T, typename E>
		static constexpr auto check_error(tl::expected<T, E> const& res) -> decltype(auto);
		template <typename T, typename E>
		static constexpr auto check_error(tl::expected<T, E>&& res) -> decltype(auto);

	protected:
		auto on_begin_first_loop() -> void override;

	private:
		EditorApplication(std::string_view name, RenderConfig config);
		~EditorApplication() override = default;

		auto create_input_actions() noexcept -> void;

		// imgui rendering
		auto draw_scene_panel() noexcept -> void;
		auto draw_object_panel() noexcept -> void;
		auto draw_scene_viewport(TextureHandle render_buf) noexcept -> void;
		auto draw_console_panel() const noexcept -> void;

		// events
		auto on_scene_opened(Scene const& scene) -> void;
		auto on_render_config_change(RenderConfig const& conf) -> void;
		auto on_mouse_leave_scene_viewport() noexcept -> void;
		auto on_mouse_enter_scene_viewport() noexcept -> void;

		// other helpers
		auto try_select_object() noexcept -> void;
		auto handle_input(InputEvent const& event) noexcept -> void override;
		auto remove_editable(Ref<SceneObject> editable) -> void;
		auto on_add_editable(Ref<SceneObject> editable) -> void;
		auto on_editable_change(Ref<SceneObject> editable, SceneObjectChangeType type) -> void;
		auto on_log_added() -> void override;
		auto get_cur_renderer() noexcept -> Renderer&;

		// event handlers
		template <auto SceneObjectChangeType, typename T>
		auto on_scene_obj_change(T data) {
			if (!m_control_state.get_cur_obj() ||
				m_control_state.get_cur_obj() != &data.obj) {
				return;
			}
			on_editable_change(data.obj, SceneObjectChangeType);
		}

		std::string m_console_text;

		std::function<void()> m_on_mouse_leave_scene_viewport_cb;
		std::function<void()> m_on_mouse_enter_scene_viewport_cb;

		// rendering
		RenderConfig m_config;
		Scene m_scene;
		Camera m_cam;

		// input handling
		std::vector<InputAction> m_input_actions;

		// rendering
		std::vector<std::unique_ptr<Renderer>> m_renderers;
		std::unique_ptr<Archive> m_archive;

		GLTextureRef m_light_icon_tex;

		struct ControlState {
			using ObjChangeCallback = std::function<void(ObserverPtr<SceneObject>)>;

			auto set_cur_obj(ObserverPtr<SceneObject> obj) noexcept -> void;
			auto get_cur_obj() const noexcept { return m_cur_obj; }
			auto register_on_obj_change(ObjChangeCallback const& callback) noexcept -> void;

			float move_sensitivity = k_init_move_sensitivity;
			float rot_sensitivity = k_init_rot_sensitivity;
			std::array<char, 1024> obj_name_buf{};
			bool is_outside_view{false};
			int cur_renderer_idx{k_default_renderer_idx};
			std::optional<glm::vec2> last_mouse_pos{std::nullopt};

			struct GizmoState {
				ImGuizmo::OPERATION op{ImGuizmo::OPERATION::TRANSLATE};
				ImGuizmo::MODE mode{ImGuizmo::MODE::WORLD};
				bool snap{false};
				glm::vec3 snap_scale{1.0};
			} gizmo_state{};

			bool unlimited_fps{true};
			bool disable_log_flush{true};

		private:
			ObserverPtr<SceneObject> m_cur_obj{nullptr};
			std::vector<ObjChangeCallback> m_obj_change_callbacks;
		} m_control_state;
	};

	template <typename T, typename E>
	constexpr auto EditorApplication::check_error(tl::expected<T, E> const& res) -> decltype(auto) {
		if (!res) {
			std::cerr << res.error() << std::endl;
			get().quit(-1);
		}
		if constexpr (!std::is_void_v<T>) {
			return res.value();
		}
	}

	template <typename T, typename E>
	constexpr auto EditorApplication::check_error(tl::expected<T, E>&& res) -> decltype(auto) {
		if (!res) {
			std::cerr << res.error() << std::endl;
			get().quit(-1);
		}
		if constexpr (!std::is_void_v<T>) {
			return std::move(res).value();
		}
	}
}
