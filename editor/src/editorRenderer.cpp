#include "include/editorRenderer.h"

#include <cassert>
#include <algorithm>

EditorRenderer::EditorRenderer(RenderConfig config)
	: Renderer{config}, m_vao{0}, m_fbo{0}, m_tex{0}, m_depth_rbo{0}, m_res{config.width, config.height} { }

EditorRenderer::~EditorRenderer() {
    if (EditorRenderer::valid()) {
        glDeleteBuffers(get_bufhandle_size(), get_bufhandles());
        glDeleteVertexArrays(1, &m_vao);
        glDeleteFramebuffers(1, &m_fbo);
        glDeleteRenderbuffers(1, &m_depth_rbo);
        glDeleteTextures(1, &m_tex);
    }
}

auto EditorRenderer::exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> {
    struct Handler {
        Handler(EditorRenderer& rend) : rend(rend) { }
        void operator()(Cmd_CameraRot const& rot) const {
            rend.get_cam().set_rotation(TransformSpace::LOCAL, rot.angles_deg);
        }
        void operator()(Cmd_CameraMove const& rot) const {
            rend.get_cam().set_position(TransformSpace::LOCAL, rot.delta);
        }
        void operator()(Cmd_CameraZoom const& zoom) const {
            rend.get_cam().set_position(TransformSpace::LOCAL, { 0, 0, zoom.delta });
        }
        EditorRenderer& rend;
    } handler{ *this };

    if (valid()) {
        std::visit(handler, cmd);
    }
    return {};
}

auto EditorRenderer::open_scene(Scene scene) noexcept -> tl::expected<void, std::string> {
    GLenum err;

    if (!valid()) {
        // set up buffers
        glGenVertexArrays(1, &m_vao);

        m_buffer_handles.resize(1);
        glGenBuffers(1, m_buffer_handles.data());

        err = glGetError();
        if (err != GL_NO_ERROR) {
            return unexpected_gl_error(err);
        }

        auto res = create_frame_buf();
        if (!res) {
            return tl::unexpected{ res.error() };
        }
    }

    // fill buffers with data
    glBindVertexArray(m_vao);
    {
        glBindBuffer(GL_ARRAY_BUFFER, get_vbo());
        {
            // gather all vertices from all objects in the scene
            std::vector<Vertex> all_vertices;
            for (auto&& obj : scene.objects()) {
                all_vertices.insert(all_vertices.end(), obj.vertices().begin(), obj.vertices().end());
            }

            glBufferData(
                GL_ARRAY_BUFFER,
                all_vertices.size() * sizeof(Vertex),
                all_vertices.data(),
                GL_STATIC_DRAW
            );
            err = glGetError();
            if (err != GL_NO_ERROR) {
                return unexpected_gl_error(err);
            }

            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void*>(offsetof(Vertex, position)));
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void*>(offsetof(Vertex, normal)));
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void*>(offsetof(Vertex, uv)));
            err = glGetError();
            if (err != GL_NO_ERROR) {
                return unexpected_gl_error(err);
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        return unexpected_gl_error(err);
    }

    m_scene = std::move(scene);

    // Set a few settings/modes in OpenGL rendering
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POLYGON_SMOOTH);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glBindVertexArray(m_vao);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        return unexpected_gl_error(err);
    }

    // Set up camera position so it has a good view of the object
    get_cam().set_transform(m_scene.get_good_cam_start());

    return {};

#undef CHECK_GL
}

auto EditorRenderer::render() noexcept -> tl::expected<void, std::string> {
    return render_internal(0);
}

auto EditorRenderer::render_buffered() noexcept -> tl::expected<RenderResultRef, std::string> {
    auto res = render_internal(m_fbo);
    if (!res) {
        return tl::unexpected{ res.error() };
    }

    auto& pixels = m_res.get_pixels();
    int w = m_res.get_width(), h = m_res.get_height();

    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    // flip vertically
    for (int y = 0; y < h / 2; ++y) {
        for (int x = 0; x < w; ++x) {
            int const frm = y * w + x;
            int const to = (h - y - 1) * w + x;
            for (int k = 0; k < 3; ++k) {
                std::swap(pixels[3 * frm + k], pixels[3 * to + k]);
            }
        }
    }

    return m_res;
}

auto EditorRenderer::render_internal(GLuint fbo) noexcept -> tl::expected<void, std::string> {
    if (!valid()) {
        return tl::unexpected{ "render() called on invalid EditorRenderer" };
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, get_config().width, get_config().height);

    auto res = m_scene.begin_draw();
    if (!res) {
        return tl::unexpected{ res.error() };
    }

	for (size_t i = 0, vbo_offset = 0; i < m_scene.objects().size(); ++i) {
        auto&& obj = m_scene.objects()[i];

        res = obj.begin_draw(get_cam());
        if (!res) {
            return tl::unexpected{ res.error() };
        }
        glDrawArrays(GL_TRIANGLES,
            static_cast<GLint>(vbo_offset),
            static_cast<GLsizei>(obj.vertices().size()));
        vbo_offset += obj.vertices().size();

        obj.end_draw();
    }

    return {};
}

auto EditorRenderer::create_frame_buf() noexcept -> tl::expected<void, std::string> {
    GLuint fbo, rbo, tex;
    GLsizei const w = get_config().width;
    GLsizei const h = get_config().height;

    // create frame buffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    {
	    // we need depth testing
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        {
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
        }
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            return unexpected_gl_error(err);
        }

        // create tex to render to
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        {
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, // RGB color format
                w, h,
                0, GL_RGB, GL_UNSIGNED_BYTE, 0
            );

            err = glGetError();
            if (err != GL_NO_ERROR) {
                return unexpected_gl_error(err);
            }

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
            GLenum draw_buffer = GL_COLOR_ATTACHMENT0;
            glDrawBuffers(1, &draw_buffer);

            err = glGetError();
            if (err != GL_NO_ERROR) {
                return unexpected_gl_error(err);
            }

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                return tl::unexpected{ "frame buffer is not valid" };
            }
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    m_fbo = fbo;
    m_depth_rbo = rbo;
    m_tex = tex;

    return {};
}
