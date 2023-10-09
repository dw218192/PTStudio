#include "renderer.h"

struct EditorRenderer : Renderer {
    EditorRenderer(RenderConfig config);
    ~EditorRenderer() override;

    [[nodiscard]] auto open_scene(Scene scene) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render() noexcept -> tl::expected<void, std::string> override;
    [[nodiscard]] auto render_buffered() noexcept -> tl::expected<RenderResultRef, std::string> override;
	[[nodiscard]] auto valid() const noexcept -> bool override { return m_vao != 0; }

private:
    auto render_internal(GLuint fbo) noexcept -> tl::expected<void, std::string>;
    auto create_frame_buf() noexcept -> tl::expected<void, std::string>;

    // use 1 vao, 1 vbo for all meshes (objects)
    GLuint m_vao;

    // buffers for rendering
    GLuint m_fbo;
    GLuint m_tex;
    GLuint m_depth_rbo;

    Scene m_scene;
    RenderResult m_res;

    [[nodiscard]] auto get_vbo() const noexcept { return m_buffer_handles[0]; }
    [[nodiscard]] auto get_bufhandle_size() const noexcept { return static_cast<GLsizei>(m_buffer_handles.size()); }
    [[nodiscard]] auto get_bufhandles() const noexcept { return m_buffer_handles.data(); }
    std::vector<GLuint> m_buffer_handles;
};