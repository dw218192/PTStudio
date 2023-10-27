#pragma once

#include "renderer.h"
#include "shader.h"
#include "glTexture.h"
#include "glFrameBuffer.h"
#include "glVertexArray.h"
#include "editorRenderer.h"

#include <unordered_map>
#include <array>
#include <singleton.h>

#include "editorResources.h"
#include "TextEditor.h"
#include "enumArray.h"

struct EditorRenderer final : Renderer, Singleton<EditorRenderer> {
friend Singleton;
	NO_COPY_MOVE(EditorRenderer);

    ~EditorRenderer() noexcept override;
    [[nodiscard]] auto init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render(View<Camera> cam) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render_buffered(View<Camera> cam) noexcept -> tl::expected<TextureHandle, std::string> override;
	[[nodiscard]] auto valid() const noexcept -> bool override { return m_valid; }
    [[nodiscard]] auto on_change_render_config(RenderConfig config) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto on_add_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto on_remove_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto draw_imgui() noexcept -> tl::expected<void, std::string> override;

    void on_object_change(ViewPtr<Object> obj) noexcept;
private:
    EditorRenderer(RenderConfig config) noexcept;

    struct PerObjectData;
    struct PerTextEditorData;
    struct PerObjectEditingData;

    [[nodiscard]] auto on_add_object_internal(Ref<PerObjectData> data, ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto render_internal(View<Camera> cam, GLuint fbo) noexcept -> tl::expected<void, std::string>;
    auto clear_render_data() -> void;

    auto commit_cur_shader_code() noexcept -> void;

    auto try_compile() noexcept -> void;
    auto draw_text_editor_header(Ref<PerTextEditorData> editor_ref) noexcept -> void;
    auto draw_text_editor(Ref<PerTextEditorData> editor_ref) noexcept -> void;
    auto draw_glsl_editor(ShaderType type, Ref<ShaderProgram> shader_ref, Ref<PerTextEditorData> editor_ref) noexcept
        -> tl::expected <void, std::string>;

    auto try_get_obj_data(ViewPtr<Object> obj) noexcept -> tl::expected<Ref<PerObjectData>, std::string>;

    ViewPtr<Scene> m_scene{ nullptr };
    GLFrameBufferRef m_render_buf{ nullptr };

    // outline drawing states
    struct {
        GLFrameBufferRef render_buf{ nullptr };
        std::array<ShaderProgramRef, std::size(vs_outline_passes)> shaders{};
        GLVertexArrayRef quad_render_data{ nullptr };
    } m_outline;

    // grid drawing states
    GLVertexArrayRef m_grid_render_data{ nullptr };
    ShaderProgramRef m_grid_shader{ nullptr };

    // default shader
    ShaderProgramRef m_default_shader{ nullptr };

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
        ShaderProgramRef shader{ nullptr };
        GLVertexArrayRef render_data{ nullptr };
        PerObjectEditingData editing_data;
    };
    std::unordered_map<ViewPtr<Object>, PerObjectData> m_obj_data;

    ViewPtr<Object> m_cur_outline_obj{ nullptr };
    bool m_valid{ false };

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
        int cur_font_size{ 16 };
        bool show_built_in_uniform{ false };
        bool show_uniform{ false };
    	bool show_input{ false };
        bool show_output{ false };
        bool show_uniform_decl{ false };
    } m_shared_editor_data;

    EArray<ShaderType, PerTextEditorData> m_shader_editor_data {};
    PerTextEditorData m_extra_func_editor_data{};
};
