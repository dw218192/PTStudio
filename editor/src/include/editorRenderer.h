#pragma once

#include "renderer.h"
#include "shader.h"
#include "glTexture.h"
#include "glFrameBuffer.h"
#include "glVertexArray.h"
#include "editorRenderer.h"

#include <unordered_map>
#include <array>

#include "editorResources.h"
#include "TextEditor.h"
#include "enumArray.h"
#include "callbackList.h"
#include "continuousGPUBufferLink.h"

namespace PTS {
	namespace Editor {
		// outline drawing states
		struct OutlineDrawingStates {
			GLFrameBufferRef render_buf{nullptr};
			std::array<ShaderProgramRef, std::size(vs_outline_passes)> shaders{};
			GLVertexArrayRef quad_render_data{nullptr};
		};

		struct GizmoSpriteData {
			GLTextureRef texture{nullptr};
			GLVertexArrayRef render_data{nullptr};
			ShaderProgramRef shader{nullptr};
		};

		// for shader editing
		// param passing here is a little weird because we have to work with
		// TextEditor
		struct PerObjectEditingData {
			std::string common_funcs;

			enum class CompilationStatus {
				UNKNOWN,
				SUCCESS,
				FAILURE
			} compilation_status = CompilationStatus::UNKNOWN;

			PerObjectEditingData();
			void set_src(ShaderType type, std::string src);
			auto get_src(ShaderType type) -> View<std::string>;
			auto get_outputs(ShaderType type) -> std::string_view;
			auto get_inputs(ShaderType type) -> std::string_view;

		private:
			EArray<ShaderType, std::string> m_shader_inputs;
			EArray<ShaderType, std::string> m_shader_outputs;
			EArray<ShaderType, std::string> m_shader_srcs; // unprocessed src
		};

		// extra object data
		struct PerObjectData {
			ShaderProgramRef shader{nullptr};
			GLVertexArrayRef render_data{nullptr};
			PerObjectEditingData editing_data{};
		};

		struct PerTextEditorData {
			TextEditor editor;
		};

		struct SharedTextEditorData {
			static constexpr int min_font_size = 12;
			static constexpr int max_font_size = 36;
			static constexpr int num_font_sizes = max_font_size - min_font_size + 1;

			// this can be generated during compile time but whatever
			inline static std::array<std::string, num_font_sizes> font_size_strs{};
			inline static std::array<ImFont*, num_font_sizes> fonts{};

			static auto get_font_size_str(int size) -> char const* {
				return font_size_strs[size - min_font_size].c_str();
			}

			static auto get_font(int size) -> ImFont* {
				return fonts[size - min_font_size];
			}

			int cur_font_size{16};
			bool show_built_in_uniform{false};
			bool show_uniform{false};
			bool show_input{false};
			bool show_output{false};
			bool show_uniform_decl{false};
		};


		struct EditorRenderer final : Renderer {
			NO_COPY_MOVE(EditorRenderer);
			EditorRenderer(RenderConfig config) noexcept;
			~EditorRenderer() noexcept override;

			[[nodiscard]] auto init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> override;
			[[nodiscard]] auto open_scene(Ref<Scene> scene) noexcept -> tl::expected<void, std::string> override;
			[[nodiscard]] auto
			render(View<Camera> cam_view) noexcept -> tl::expected<TextureHandle, std::string> override;
			[[nodiscard]] auto valid() const noexcept -> bool override { return m_valid; }
			[[nodiscard]] auto draw_imgui() noexcept -> tl::expected<void, std::string> override;
			void on_selected_editable_change(ObserverPtr<SceneObject> editable) noexcept;

		protected:
			[[nodiscard]] auto on_change_render_config() noexcept -> tl::expected<void, std::string> override;

		private:
			[[nodiscard]] auto on_add_object_internal(PerObjectData& data,
			                                          RenderableObject const& obj) noexcept -> tl::expected<
				void, std::string>;
			[[nodiscard]] auto draw_outline(View<Camera> cam_view,
			                                View<RenderableObject> obj) -> tl::expected<void, std::string>;
			auto clear_render_data() -> void;

			auto commit_cur_shader_code() noexcept -> void;

			auto try_compile() noexcept -> void;
			auto draw_text_editor_header(Ref<PerTextEditorData> editor_ref) noexcept -> void;
			auto draw_text_editor(Ref<PerTextEditorData> editor_ref) const noexcept -> void;
			auto draw_glsl_editor(ShaderType type, Ref<ShaderProgram> shader_ref,
			                      Ref<PerTextEditorData> editor_ref) noexcept
				-> tl::expected<void, std::string>;

			auto try_get_obj_data(View<RenderableObject> obj) noexcept -> tl::expected<Ref<PerObjectData>, std::string>;

			// callbacks
			auto on_add_obj(Ref<SceneObject> obj) noexcept -> void;
			auto on_remove_obj(Ref<SceneObject> obj) noexcept -> void;
			auto update_light(Light const& light) -> void;

			Callback<void(Ref<SceneObject>)> m_on_add_obj
				{[this](Ref<SceneObject> data) { this->on_add_obj(data); }};
			Callback<void(Ref<SceneObject>)> m_on_remove_obj
				{[this](Ref<SceneObject> data) { this->on_remove_obj(data); }};

			DECL_FIELD_EVENT_MEMBERS(on_light_color_change, Light, Light::FieldTag::COLOR);
			DECL_FIELD_EVENT_MEMBERS(on_light_intensity_change, Light, Light::FieldTag::INTENSITY);
			DECL_FIELD_EVENT_MEMBERS(on_obj_world_trans_change, SceneObject, SceneObject::FieldTag::WORLD_TRANSFORM);
			DECL_FIELD_EVENT_MEMBERS(on_obj_local_trans_change, SceneObject, SceneObject::FieldTag::LOCAL_TRANSFORM);

			// fields
			ObserverPtr<Scene> m_scene{nullptr};
			GLFrameBufferRef m_render_buf{nullptr};
			OutlineDrawingStates m_outline; // this is lazily initialized
			// gizmo sprite drawing states, should be initialized in init()
			GizmoSpriteData m_light_gizmo_data;
			// grid drawing states, should be initialized in init()
			GLVertexArrayRef m_grid_render_data{nullptr};
			ShaderProgramRef m_grid_shader{nullptr};
			// default shader, should be initialized in init()
			ShaderProgramRef m_default_shader{nullptr};
			// light data
			ContinuousGPUBufferLink<LightData, Light, GLBufferRef, k_maxLights> m_light_data_link{};
			std::unordered_map<ViewPtr<RenderableObject>, PerObjectData> m_obj_data;
			ViewPtr<RenderableObject> m_cur_outline_obj{nullptr};
			bool m_valid{false};
			SharedTextEditorData m_shared_editor_data;

			EArray<ShaderType, PerTextEditorData> m_shader_editor_data{};
			PerTextEditorData m_extra_func_editor_data{};
		};
	}
}
