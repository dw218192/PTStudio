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

struct EditorRenderer final : Renderer, Singleton<EditorRenderer> {
friend Singleton;

    ~EditorRenderer() noexcept override;
    [[nodiscard]] auto init() noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render(View<Camera> cam) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render_buffered(View<Camera> cam) noexcept -> tl::expected<TextureHandle, std::string> override;
	[[nodiscard]] auto valid() const noexcept -> bool override { return m_valid; }
    [[nodiscard]] auto on_change_render_config(RenderConfig const& config) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto on_add_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto on_remove_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto draw_imgui(ViewPtr<Application> app) noexcept -> tl::expected<void, std::string> override;

    void on_object_change(ViewPtr<Object> obj) noexcept;
private:
    EditorRenderer(RenderConfig config) noexcept;

    struct PerObjectData;
    [[nodiscard]] auto on_add_object_internal(PerObjectData& data, ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto render_internal(View<Camera> cam, GLuint fbo) noexcept -> tl::expected<void, std::string>;
    void clear_render_data();


	ViewPtr<Scene> m_scene;
    GLFrameBufferRef m_render_buf;

    // outline drawing states
    struct {
        GLFrameBufferRef render_buf{ nullptr };
        std::array<ShaderProgramRef, std::size(vs_outline_passes)> shaders{};
        GLVertexArrayRef quad_render_data{ nullptr };
    } m_outline;

    // grid drawing states
    GLVertexArrayRef m_grid_render_data;
    ShaderProgramRef m_grid_shader;

    // default shader
    ShaderProgramRef m_default_shader;

    // extra object data
    struct PerObjectData {
        ShaderProgramRef shader{ nullptr };
        GLVertexArrayRef render_data{ nullptr };
    };
    std::unordered_map<ViewPtr<Object>, PerObjectData> m_render_data;

    ViewPtr<Object> m_cur_outline_obj{ nullptr };
    bool m_valid{ false };
};
