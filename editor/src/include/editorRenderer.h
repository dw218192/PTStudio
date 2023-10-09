#include "renderer.h"

struct EditorRenderer : Renderer {
    EditorRenderer(RenderConfig const& config) noexcept;
    ~EditorRenderer() override;

    [[nodiscard]] auto open_scene(Scene scene) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render() noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render_buffered() noexcept -> tl::expected<RenderResultRef, std::string> override;
	[[nodiscard]] auto valid() const noexcept -> bool override { return m_vao != 0; }

private:
    struct RenderBuf {
        // buffers for rendering
        GLuint fbo = 0;
        GLuint tex = 0;
        GLuint depth_rbo = 0;
    };

    [[nodiscard]] auto render_internal(GLuint fbo) noexcept -> tl::expected<void, std::string>;
    [[nodiscard]] auto create_frame_buf() noexcept -> tl::expected<RenderBuf, std::string>;
    [[nodiscard]] auto create_default_shader() noexcept -> tl::expected<ShaderProgram, std::string>;
    
    [[nodiscard]] auto get_vbo() const noexcept { return m_buffer_handles[0]; }
    [[nodiscard]] auto get_bufhandle_size() const noexcept { return static_cast<GLsizei>(m_buffer_handles.size()); }
    [[nodiscard]] auto get_bufhandles() const noexcept { return m_buffer_handles.data(); }

    // use 1 vao, 1 vbo for all meshes (objects)
    GLuint m_vao = 0;
    RenderBuf m_rend_buf;
    Scene m_scene;
    RenderResult m_res;
    std::vector<GLuint> m_buffer_handles;

    // shader
    ShaderProgram m_editor_shader;
};