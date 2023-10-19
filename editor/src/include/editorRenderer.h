#pragma once

#include "renderer.h"
#include "shader.h"
#include "glTexture.h"
#include "glFrameBuffer.h"
#include "glRenderBuffer.h"
#include "glBuffer.h"

#include <optional>
#include <unordered_map>

#include "editorRenderer.h"

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
    struct RenderBufferData;
    struct ObjectRenderData;

    [[nodiscard]] auto on_add_object_internal(ObjectRenderData& data, ConstObjectHandle obj) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto render_internal(Camera const& cam, GLuint fbo) noexcept -> tl::expected<void, std::string>;
	[[nodiscard]] auto create_render_buf() noexcept -> tl::expected<void, std::string>;

    void clear_render_data();

    struct RenderBufferData {
        GLFrameBufferRef fbo;
        GLRenderBufferRef rbo;
        GLTextureRef tex;
        RenderBufferData(GLFrameBufferRef fbo, GLRenderBufferRef rbo, GLTextureRef tex_data) noexcept
            : fbo{std::move(fbo)}, rbo{std::move(rbo)}, tex{std::move(tex_data)} {}
    };
    struct ObjectRenderData {
        GLBufferRef vao;
        GLBufferRef vbo;
        GLsizei vertex_count{ 0 };
    };

    std::optional<RenderBufferData> m_render_buf;

    ObjectRenderData m_grid_render_data;
    std::unordered_map<ConstObjectHandle, ObjectRenderData> m_render_data;

	ConstObjectHandle m_cur_outline_obj = nullptr;
    bool m_valid = false;


    ShaderProgramRef m_editor_shader;
    ShaderProgramRef m_grid_shader;
    ShaderProgramRef m_outline_shader;
};
