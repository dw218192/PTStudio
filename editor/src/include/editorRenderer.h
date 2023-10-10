#include "renderer.h"
#include <array>

struct EditorRenderer : Renderer {
    EditorRenderer(RenderConfig const& config) noexcept;
    ~EditorRenderer() override;

    [[nodiscard]] auto open_scene(Scene scene) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render() noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render_buffered() noexcept -> tl::expected<RenderResultRef, std::string> override;
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

    [[nodiscard]] auto render_internal(GLuint fbo) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto create_frame_buf() noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto get_buf(BufIndex idx) noexcept -> GLuint& { return m_buffer_handles[idx]; }
    [[nodiscard]] auto get_vao(VAOIndex idx) noexcept -> GLuint& { return m_vao_handles[idx]; }
    [[nodiscard]] auto get_bufs() const noexcept { return m_buffer_handles.data(); }
    [[nodiscard]] auto get_vaos() const noexcept { return m_vao_handles.data(); }

    std::array<GLuint, VAO_COUNT> m_vao_handles;
    std::array<GLuint, BUF_COUNT> m_buffer_handles;

    // use 1 vao, 1 vbo for all meshes (objects)
    Scene m_scene;
    RenderResult m_res;
    struct {
        GLuint fbo;
        GLuint tex;
        GLuint rbo;
    } m_render_buf;

    // shader
    bool m_valid = false;
    ShaderProgram m_editor_shader;

    unsigned m_grid_ebo_count = 0;
    ShaderProgram m_grid_shader;
};