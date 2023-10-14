#include "include/editorRenderer.h"
#include "include/application.h"
#include "include/editorResources.h"

#include <cassert>
#include <algorithm>

constexpr float k_grid_dim = 100.0f;
constexpr float k_grid_spacing = 1.0f;
constexpr glm::vec3 k_clear_color = glm::vec3{ 0 };

EditorRenderer::EditorRenderer(RenderConfig const& config) noexcept
	: Renderer{config} {}

EditorRenderer::~EditorRenderer() {
    if (EditorRenderer::valid()) {
        glDeleteBuffers(BufIndex::BUF_COUNT, get_bufs());
        glDeleteVertexArrays(VAOIndex::VAO_COUNT, get_vaos());

        if(m_render_buf) {
            glDeleteFramebuffers(1, &m_render_buf->fbo);
            glDeleteRenderbuffers(1, &m_render_buf->rbo);
            glDeleteTextures(1, &m_render_buf->tex);
        }
    }
}

auto EditorRenderer::exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> {
    struct Handler {
        Handler(EditorRenderer& rend) : rend(rend) { }
        void operator()(Cmd_CameraRot const& cmd) const {
            Application::get_cam().set_rotation(TransformSpace::LOCAL, cmd.angles_deg);
        }
        void operator()(Cmd_CameraMove const& cmd) const {
			Application::get_cam().set_position(TransformSpace::LOCAL, cmd.delta);
        }
        void operator()(Cmd_CameraZoom const& cmd) const {
            Application::get_cam().set_position(TransformSpace::LOCAL, { 0, 0, cmd.delta });
        }
        void operator()(Cmd_ChangeRenderConfig const& cmd) {
            if (!cmd.config.is_valid() || cmd.config == rend.m_config) {
                return;
            }
            rend.m_config = cmd.config;
            auto res = rend.create_render_buf();
            if (!res) {
                err = res.error();
            }

            Application::get_cam().set_viewport(cmd.config.width, cmd.config.height);
            Application::get_cam().set_fov(rend.m_config.fovy);
        }

        EditorRenderer& rend;
        std::string err;
    } handler{ *this };

    if (valid()) {
        std::visit(handler, cmd);
        if (!handler.err.empty()) {
            return tl::unexpected{ handler.err };
        }
    }
    return {};
}

auto EditorRenderer::open_scene(Scene const& scene) noexcept -> tl::expected<void, std::string> {
    auto create_grid = [this](float grid_dim, float spacing) -> tl::expected<void, std::string> {
        std::vector<glm::vec3> vertices;
        std::vector<unsigned> indices;

        float const half_dim = grid_dim / 2.0f;
        for (float x = -half_dim; x <= half_dim; x += spacing) {
            vertices.emplace_back(x, 0.0f, -half_dim);
            vertices.emplace_back(x, 0.0f, half_dim);
        }
        for (float z = -half_dim; z <= half_dim; z += spacing) {
            vertices.emplace_back(-half_dim, 0.0f, z);
            vertices.emplace_back(half_dim, 0.0f, z);
        }
        for (unsigned i = 0; i < vertices.size(); ++i) {
            indices.push_back(i);
        }

        glBindVertexArray(get_vao(VAOIndex::GRID_VAO));
        {
            glBindBuffer(GL_ARRAY_BUFFER, get_buf(BufIndex::GRID_VBO));
            {
                glBufferData(
                    GL_ARRAY_BUFFER,
                    vertices.size() * sizeof(glm::vec3),
                    vertices.data(),
                    GL_STATIC_DRAW
                );
                auto err = glGetError();
                if (err != GL_NO_ERROR) {
                    return unexpected_gl_error(err);
                }

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
                err = glGetError();
                if (err != GL_NO_ERROR) {
                    return unexpected_gl_error(err);
                }
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, get_buf(BufIndex::GRID_EBO));
            {
                glBufferData(
                    GL_ELEMENT_ARRAY_BUFFER,
                    indices.size() * sizeof(GLuint),
                    indices.data(),
                    GL_STATIC_DRAW
                );
                auto err = glGetError();
                if (err != GL_NO_ERROR) {
                    return unexpected_gl_error(err);
                }
            }
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        glBindVertexArray(0);
        auto err = glGetError();
        if (err != GL_NO_ERROR) {
            return unexpected_gl_error(err);
        }

        m_grid_ebo_count = indices.size();

        // set invariant uniforms
        m_grid_shader.use();
        auto res = m_grid_shader.set_uniform(k_uniform_half_grid_dim, half_dim);
        if (!res) {
            return res;
        }
        m_grid_shader.unuse();

        return {};
    };

    auto init = [this, &create_grid]() -> tl::expected<void, std::string> {
        // set up object buffers
        glGenVertexArrays(VAOIndex::VAO_COUNT, m_vao_handles.data());
        glGenBuffers(BufIndex::BUF_COUNT, m_buffer_handles.data());

        auto err = glGetError();
        if (err != GL_NO_ERROR) {
            return unexpected_gl_error(err);
        }
        // set up shader
        auto shader_res = ShaderProgram::from_srcs(vs_obj_src, ps_obj_src);
        if (!shader_res) {
            return tl::unexpected{ shader_res.error() };
        } else {
            m_editor_shader = std::move(shader_res.value());
        }

        shader_res = ShaderProgram::from_srcs(vs_grid_src, ps_grid_src);
        if (!shader_res) {
            return tl::unexpected{ shader_res.error() };
        } else {
            m_grid_shader = std::move(shader_res.value());
        }

        auto res = create_grid(k_grid_dim, k_grid_spacing);
        if (!res) {
            return res;
        }

        return {};
    };

    if (!valid()) {
        auto res = init();
        if(!res) {
            return res;
        }
        m_valid = true;
    }

    // fill buffers with data
    glBindVertexArray(get_vao(VAOIndex::OBJ_VAO));
    {
        glBindBuffer(GL_ARRAY_BUFFER, get_buf(BufIndex::OBJ_VBO));
        {
            // gather all vertices from all objects in the scene
            std::vector<Vertex> all_vertices;
            for (auto&& obj : scene) {
                all_vertices.insert(all_vertices.end(), obj.get_vertices().begin(), obj.get_vertices().end());
            }

            glBufferData(
                GL_ARRAY_BUFFER,
                all_vertices.size() * sizeof(Vertex),
                all_vertices.data(),
                GL_STATIC_DRAW
            );

            auto err = glGetError();
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
    auto err = glGetError();
    if (err != GL_NO_ERROR) {
        return unexpected_gl_error(err);
    }

    // Set a few settings/modes in OpenGL rendering
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POLYGON_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    err = glGetError();
    if (err != GL_NO_ERROR) {
        return unexpected_gl_error(err);
    }

    return {};

#undef CHECK_GL
}

auto EditorRenderer::render() noexcept -> tl::expected<void, std::string> {
    return render_internal(0);
}

auto EditorRenderer::render_buffered() noexcept -> tl::expected<TextureRef, std::string> {
    if (!m_render_buf) {
        auto res = create_render_buf();
        if (!res) {
            return tl::unexpected{ res.error() };
        }
    }

    auto res = render_internal(m_render_buf->fbo);
    if (!res) {
        return tl::unexpected{ res.error() };
    }

    return std::cref(m_render_buf->tex_data);
}

auto EditorRenderer::render_internal(GLuint fbo) noexcept -> tl::expected<void, std::string> {
    if (!valid()) {
        return tl::unexpected{ "render_internal() called on invalid EditorRenderer" };
    } else if (!m_grid_shader.valid()) {
        return tl::unexpected{ "render_internal() called on EditorRenderer with invalid grid shader" };
    } else if (!m_editor_shader.valid()) {
        return tl::unexpected{ "render_internal() called on EditorRenderer with invalid editor shader" };
    }

    auto&& cam = Application::get_cam();
    auto&& scene = Application::get_scene();

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClearColor(k_clear_color.x, k_clear_color.y, k_clear_color.z, 1.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glViewport(0, 0, get_config().width, get_config().height);

    glBindVertexArray(get_vao(VAOIndex::OBJ_VAO));
    // render objects
    m_editor_shader.use();
    {
        auto res = m_editor_shader.set_uniform(k_uniform_light_pos, scene.get_good_light_pos());
        if (!res) {
            return res;
        }
        res = m_editor_shader.set_uniform(k_uniform_view, cam.get_view());
        if (!res) {
            return res;
        }
        res = m_editor_shader.set_uniform(k_uniform_projection, cam.get_projection());
        if (!res) {
            return res;
        }

        size_t vbo_offset = 0;
        for (const auto& obj : scene) {
            res = m_editor_shader.set_uniform(k_uniform_model, obj.get_transform().get_matrix());
            if (!res) {
                return res;
            }

            glDrawArrays(GL_TRIANGLES,
                static_cast<GLint>(vbo_offset),
                static_cast<GLsizei>(obj.get_vertices().size()));
            vbo_offset += obj.get_vertices().size();

            auto err = glGetError();
            if (err != GL_NO_ERROR) {
                return unexpected_gl_error(err);
            }
        }
    }
    m_editor_shader.unuse();

    // render grid
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(get_vao(VAOIndex::GRID_VAO));

    m_grid_shader.use();
    {
        auto res = m_grid_shader.set_uniform(k_uniform_view, cam.get_view());
        if (!res) {
            return res;
        }
        res = m_grid_shader.set_uniform(k_uniform_projection, cam.get_projection());
        if (!res) {
            return res;
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, get_buf(BufIndex::GRID_EBO));
        glDrawElements(GL_LINES, m_grid_ebo_count, GL_UNSIGNED_INT, nullptr);
    }
    m_grid_shader.unuse();

    auto err = glGetError();
    if (err != GL_NO_ERROR) {
        return unexpected_gl_error(err);
    }
    glDisable(GL_BLEND);

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return {};
}

auto EditorRenderer::create_render_buf() noexcept -> tl::expected<void, std::string> {
    if (m_render_buf) {
        glDeleteFramebuffers(1, &m_render_buf->fbo);
        glDeleteRenderbuffers(1, &m_render_buf->rbo);
        glDeleteTextures(1, &m_render_buf->tex);
    }

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
                0, GL_RGB, GL_UNSIGNED_BYTE, nullptr
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
                return tl::unexpected{ std::string { "frame buffer is not valid, status = " } + 
                    std::to_string(glCheckFramebufferStatus(GL_FRAMEBUFFER)) };
            }
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_render_buf = RenderBufferData{ fbo, tex, rbo, Texture{ static_cast<unsigned>(w), static_cast<unsigned>(h), tex } };
    return {};
}
