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

    [[nodiscard]] auto on_add_object_internal(Ref<PerObjectData> data, ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto render_internal(View<Camera> cam, GLuint fbo) noexcept -> tl::expected<void, std::string>;
    void clear_render_data();

    void commit_cur_shader_code() noexcept;
    auto draw_glsl_editor(ShaderType type, Ref<ShaderProgram> shader, PerTextEditorData& editor) noexcept
        -> tl::expected <void, std::string>;
    void preprocess_shader_code(ShaderType type, std::string& main_src);

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
    struct PerObjectEditingData {
        EArray<ShaderType, std::string> shader_srcs;
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
        static constexpr char const* font_size_mul_strs[] = { "1x", "2x", "4x" };

        char const* cur_font_size_mul_str { font_size_mul_strs[0] };
        TextEditor editor{ };

        auto get_font_size_mul() const {
            if (!cur_font_size_mul_str) {
                return 1.0f;
            }
            return static_cast<float>(cur_font_size_mul_str[0] - '0');
        }
    };
    EArray<ShaderType, PerTextEditorData> m_shader_editor_data {};

};
