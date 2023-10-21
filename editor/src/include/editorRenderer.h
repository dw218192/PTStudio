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

struct EditorRenderer : Renderer {
    EditorRenderer(RenderConfig const& config) noexcept;
    ~EditorRenderer() noexcept override;
    [[nodiscard]] auto init() noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto open_scene(Scene const& scene) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render(Camera const& cam) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render_buffered(Camera const& cam) noexcept -> tl::expected<TextureHandle, std::string> override;
	[[nodiscard]] auto valid() const noexcept -> bool override { return m_valid; }
    [[nodiscard]] auto on_change_render_config(RenderConfig const& config) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto on_add_object(ConstObjectHandle obj) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto on_remove_object(ConstObjectHandle obj) noexcept -> tl::expected<void, std::string> override;
    void on_object_change(ConstObjectHandle obj) noexcept;
private:
    [[nodiscard]] auto on_add_object_internal(GLVertexArrayRef& data, ConstObjectHandle obj) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto render_internal(Camera const& cam, GLuint fbo) noexcept -> tl::expected<void, std::string>;
    void clear_render_data();

    GLFrameBufferRef m_render_buf;

    struct {
        GLFrameBufferRef render_buf{ nullptr };
        std::array<ShaderProgramRef, std::size(vs_outline_passes)> shaders{};
        GLVertexArrayRef quad_render_data{ nullptr };
    } m_outline;

    GLVertexArrayRef m_grid_render_data;
    std::unordered_map<ConstObjectHandle, GLVertexArrayRef> m_render_data;

	ConstObjectHandle m_cur_outline_obj{ nullptr };
    bool m_valid{ false };

    ShaderProgramRef m_editor_shader;
    ShaderProgramRef m_grid_shader;
};
