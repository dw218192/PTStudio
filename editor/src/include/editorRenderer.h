#pragma once

#include "renderer.h"
#include "shader.h"
#include <array>
#include <optional>

#include "glTexture.h"

struct EditorRenderer : Renderer {
    EditorRenderer(RenderConfig const& config) noexcept;
    ~EditorRenderer() override;

    [[nodiscard]] auto open_scene(Scene const& scene) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render(Camera const& cam) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render_buffered(Camera const& cam) noexcept -> tl::expected<TextureRef, std::string> override;
	[[nodiscard]] auto valid() const noexcept -> bool override { return m_valid; }

private:
    enum BufIndex {
        OBJ_VBO,
        GRID_VBO,
        GRID_EBO,
        BUF_COUNT
    };
    enum VAOIndex {
        OBJ_VAO,
        GRID_VAO,
        VAO_COUNT
    };

    [[nodiscard]] auto render_internal(Camera const& cam, GLuint fbo) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto create_render_buf() noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto get_buf(BufIndex idx) noexcept -> GLuint& { return m_buffer_handles[idx]; }
    [[nodiscard]] auto get_vao(VAOIndex idx) noexcept -> GLuint& { return m_vao_handles[idx]; }
    [[nodiscard]] auto get_bufs() const noexcept { return m_buffer_handles.data(); }
    [[nodiscard]] auto get_vaos() const noexcept { return m_vao_handles.data(); }

    std::array<GLuint, VAO_COUNT> m_vao_handles;
    std::array<GLuint, BUF_COUNT> m_buffer_handles;

    // use 1 vao, 1 vbo for all meshes (objects)
    struct RenderBufferData {
        GLuint fbo;
        GLuint tex;
        GLuint rbo;
        GLTexture tex_data;
        RenderBufferData(GLuint fbo, GLuint tex, GLuint rbo, GLTexture tex_data) noexcept
            : fbo{fbo}, tex{tex}, rbo{rbo}, tex_data{std::move(tex_data)} {}
    };
    
    std::optional<RenderBufferData> m_render_buf;

    // shader for all objects in the editor
    bool m_valid = false;
    ShaderProgram m_editor_shader;
    // stuff for the grid
    unsigned m_grid_ebo_count = 0;
    ShaderProgram m_grid_shader;
};
