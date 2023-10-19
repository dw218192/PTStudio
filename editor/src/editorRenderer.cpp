#include "include/editorRenderer.h"
#include "include/application.h"
#include "include/editorResources.h"

#include <cassert>
#include <algorithm>

constexpr auto k_grid_dim = 100.0f;
constexpr auto k_grid_spacing = 1.0f;
constexpr auto k_clear_color = glm::vec3{ 0 };
constexpr auto k_outline_scale = 1.02f;
constexpr auto k_outline_color = glm::vec3{ 1, 0, 0 };

EditorRenderer::EditorRenderer(RenderConfig const& config) noexcept
	: Renderer{config} {}

EditorRenderer::~EditorRenderer() {
    if (EditorRenderer::valid()) {
        std::vector<GLuint> vaos, vbos;
        vaos.emplace_back(m_grid_render_data.vao);
        vbos.emplace_back(m_grid_render_data.vbo);
        for(auto&& [_, data] : m_render_data) {
            vaos.emplace_back(data.vao);
            vbos.emplace_back(data.vbo);
        }

        glDeleteBuffers(vbos.size(), vbos.data());
        glDeleteVertexArrays(vaos.size(), vaos.data());

        if(m_render_buf) {
            glDeleteFramebuffers(1, &m_render_buf->fbo);
            glDeleteRenderbuffers(1, &m_render_buf->rbo);
        }
    }
}

auto EditorRenderer::init() noexcept -> tl::expected<void, std::string> {
    auto create_grid = [this](float grid_dim, float spacing) -> tl::expected<void, std::string> {
        std::vector<glm::vec3> vertices;
        float const half_dim = grid_dim / 2.0f;
        for (float x = -half_dim; x <= half_dim; x += spacing) {
            vertices.emplace_back(x, 0.0f, -half_dim);
            vertices.emplace_back(x, 0.0f, half_dim);
        }
        for (float z = -half_dim; z <= half_dim; z += spacing) {
            vertices.emplace_back(-half_dim, 0.0f, z);
            vertices.emplace_back(half_dim, 0.0f, z);
        }

        glGenVertexArrays(1, &m_grid_render_data.vao);
        glGenBuffers(1, &m_grid_render_data.vbo);
        m_grid_render_data.vertex_count = vertices.size();

        glBindVertexArray(m_grid_render_data.vao);
        {
            glBindBuffer(GL_ARRAY_BUFFER, m_grid_render_data.vbo);
            {
                glBufferData(
                    GL_ARRAY_BUFFER,
                    vertices.size() * sizeof(glm::vec3),
                    vertices.data(),
                    GL_STATIC_DRAW
                );
                CHECK_GL_ERROR();

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
                CHECK_GL_ERROR();
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        glBindVertexArray(0);

        CHECK_GL_ERROR();

        // set invariant uniforms
        m_grid_shader->use();
        TL_CHECK_FWD(m_grid_shader->set_uniform(k_uniform_half_grid_dim, half_dim));
        m_grid_shader->unuse();

        return {};
    };

	if (valid()) {
        return {};
    }

	// set up shaders
    auto shader_res = ShaderProgram::from_srcs(vs_obj_src, ps_obj_src);
    if (!shader_res) {
        return TL_ERROR(shader_res.error());
    } else {
        m_editor_shader = std::move(shader_res.value());
    }

    shader_res = ShaderProgram::from_srcs(vs_grid_src, ps_grid_src);
    if (!shader_res) {
        return TL_ERROR(shader_res.error());
    } else {
        m_grid_shader = std::move(shader_res.value());
    }

    shader_res = ShaderProgram::from_srcs(vs_outline_src, ps_outline_src);
    if (!shader_res) {
        return TL_ERROR(shader_res.error());
    } else {
        m_outline_shader = std::move(shader_res.value());
    }

	m_outline_shader->use();
    {
        TL_CHECK_FWD(m_outline_shader->set_uniform(k_uniform_outline_color, k_outline_color));
    }
    m_outline_shader->unuse();

    TL_CHECK_FWD(create_grid(k_grid_dim, k_grid_spacing));

    m_valid = true;
    return {};
}

auto EditorRenderer::open_scene(Scene const& scene) noexcept -> tl::expected<void, std::string> {
    m_cur_outline_obj = nullptr;
    clear_render_data();

    std::vector<GLuint> vaos(scene.size()), vbos(scene.size());
    // set up object buffers
    glGenVertexArrays(scene.size(), vaos.data());
    glGenBuffers(scene.size(), vbos.data());

    CHECK_GL_ERROR();

    for (auto [it, i] = std::tuple { scene.begin(), 0 }; it != scene.end(); ++it, ++i) {
        auto obj = *it;
        auto&& data = m_render_data[obj];
        data.vao = vaos[i];
        data.vbo = vbos[i];
        data.vertex_count = obj->get_vertices().size();

        auto res = on_add_object_internal(data, obj);
        if (!res) return res;
    }

    // Set a few settings/modes in OpenGL rendering
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POLYGON_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_STENCIL_TEST);
    glClearStencil(0);
    CHECK_GL_ERROR();

    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    CHECK_GL_ERROR();

    return {};
}

auto EditorRenderer::render(Camera const& cam) noexcept -> tl::expected<void, std::string> {
    return render_internal(cam, 0);
}

auto EditorRenderer::render_buffered(Camera const& cam) noexcept -> tl::expected<TextureHandle, std::string> {
    if (!m_render_buf) {
        auto res = create_render_buf();
        if (!res) {
            return TL_ERROR(res.error());
        }
    }

    auto res = render_internal(cam, m_render_buf->fbo);
    if (!res) {
        return TL_ERROR(res.error());
    }

    return m_render_buf->tex_data.get();
}

auto EditorRenderer::on_change_render_config(RenderConfig const& config) noexcept -> tl::expected<void, std::string> {
    if (!config.is_valid() || config == m_config) {
        return {};
    }
    m_config = config;
    auto res = create_render_buf();
    if (!res) {
        return res;
    }
    return {};
}

auto EditorRenderer::on_add_object(ConstObjectHandle obj) noexcept -> tl::expected<void, std::string> {
    if (!obj) {
        return {};
    }

	// fill buffers with data
    auto&& data = m_render_data[obj];
    glGenVertexArrays(1, &data.vao);
    glGenBuffers(1, &data.vbo);
    data.vertex_count = obj->get_vertices().size();

    auto res = on_add_object_internal(data, obj);
    if(!res) {
        return res;
    }

    return {};
}

auto EditorRenderer::on_remove_object(ConstObjectHandle obj) noexcept -> tl::expected<void, std::string> {
    auto it = m_render_data.find(obj);
    if(it == m_render_data.end()) {
        return TL_ERROR("on_remove_object: obj not found");
    }

    glDeleteBuffers(1, &(it->second.vbo));
    glDeleteVertexArrays(1, &(it->second.vao));
    m_render_data.erase(obj);

    return {};
}

void EditorRenderer::on_object_change(ConstObjectHandle obj) noexcept {
    m_cur_outline_obj = obj;
}

auto EditorRenderer::on_add_object_internal(ObjectRenderData& data, ConstObjectHandle obj) noexcept -> tl::expected<void, std::string> {
    glBindVertexArray(data.vao);
    glBindBuffer(GL_ARRAY_BUFFER, data.vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        obj->get_vertices().size() * sizeof(Vertex),
        obj->get_vertices().data(),
        GL_STATIC_DRAW
    );

    CHECK_GL_ERROR();

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void*>(offsetof(Vertex, position)));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void*>(offsetof(Vertex, uv)));

    CHECK_GL_ERROR();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return {};

}

auto EditorRenderer::render_internal(Camera const& cam, GLuint fbo) noexcept -> tl::expected<void, std::string> {
    auto draw_obj = [this](ConstObjectHandle obj) -> tl::expected<void, std::string> {
        auto it = m_render_data.find(obj);
        if (it == m_render_data.end()) {
            return  TL_ERROR("obj not found in render data");
        }
        glBindVertexArray(it->second.vao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(obj->get_vertices().size()));

        CHECK_GL_ERROR();
        return {};
    };


    if (!valid()) {
        return TL_ERROR( "invalid EditorRenderer");
    } else if (!m_grid_shader->valid()) {
        return TL_ERROR("invalid grid shader");
    } else if (!m_editor_shader->valid()) {
        return TL_ERROR("invalid editor shader");
    }

    auto&& scene = Application::get_scene();

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClearColor(k_clear_color.x, k_clear_color.y, k_clear_color.z, 1.0f);
    glClearDepth(1.0f);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(0, 0, static_cast<GLsizei>(get_config().width), static_cast<GLsizei>(get_config().height));

    // render objects
    // disable stencil by default
    glStencilMask(0);
	m_editor_shader->use();
    {
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_light_pos, scene.get_good_light_pos()));
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_view, cam.get_view()));
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_projection, cam.get_projection()));

        for (auto obj : scene) {
            TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_model, obj->get_transform().get_matrix()));
            TL_CHECK_FWD(draw_obj(obj));
        }
    }
    m_editor_shader->unuse();

    // draw outlined obj if some obj is selected
    // this renders that obj 3 times which is bad
    // TODO: optimize if too slow?
    if (m_cur_outline_obj) {
        m_outline_shader->use();
        {
            glDisable(GL_DEPTH_TEST);
            glStencilOp(GL_REPLACE, GL_KEEP, GL_KEEP);
            glStencilFunc(GL_NEVER, 1, 0xff);
            glStencilMask(0xff);

            TL_CHECK_FWD(m_outline_shader->set_uniform(k_uniform_model, m_cur_outline_obj->get_transform().get_matrix()));
            TL_CHECK_FWD(m_outline_shader->set_uniform(k_uniform_view, cam.get_view()));
            TL_CHECK_FWD(m_outline_shader->set_uniform(k_uniform_projection, cam.get_projection()));

            TL_CHECK_FWD(m_outline_shader->set_uniform(k_uniform_scale_factor, glm::mat4{1}));
        	TL_CHECK_FWD(draw_obj(m_cur_outline_obj));

            glStencilFunc(GL_NOTEQUAL, 1, 0xff);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

        	TL_CHECK_FWD(m_outline_shader->set_uniform(k_uniform_scale_factor, glm::scale(glm::vec3{ k_outline_scale })));
            TL_CHECK_FWD(draw_obj(m_cur_outline_obj));

            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glEnable(GL_DEPTH_TEST);
        }
        m_outline_shader->unuse();
    }

    // render grid
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(m_grid_render_data.vao);
    m_grid_shader->use();
    {
        TL_CHECK_FWD(m_grid_shader->set_uniform(k_uniform_view, cam.get_view()));
        TL_CHECK_FWD(m_grid_shader->set_uniform(k_uniform_projection, cam.get_projection()));

    	glDrawArrays(GL_LINES, 0, m_grid_render_data.vertex_count);
        CHECK_GL_ERROR();
    }
    m_grid_shader->unuse();
    glBindVertexArray(0);

    glDisable(GL_BLEND);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return {};
}

auto EditorRenderer::create_render_buf() noexcept -> tl::expected<void, std::string> {
    if (m_render_buf) {
        glDeleteFramebuffers(1, &m_render_buf->fbo);
        glDeleteRenderbuffers(1, &m_render_buf->rbo);
    }

    GLuint fbo, rbo;
    GLTextureRef tex;

    auto w = static_cast<GLsizei>(get_config().width);
    auto h = static_cast<GLsizei>(get_config().height);

    // create frame buffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    {
	    // create both depth and stencil buffers
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        {
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8_OES, w, h);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
        }
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);

        CHECK_GL_ERROR();

        // create tex to render to
        auto res = GLTexture::create(w, h);
        if (!res) {
            return TL_ERROR(res.error());
        }

    	tex = std::move(res.value());
        tex->bind();
        {
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex->handle(), 0);
            GLenum draw_buffer = GL_COLOR_ATTACHMENT0;
            glDrawBuffers(1, &draw_buffer);

            CHECK_GL_ERROR();

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                return TL_ERROR(std::string{ "frame buffer is not valid, status = " } +
                    std::to_string(glCheckFramebufferStatus(GL_FRAMEBUFFER)));
            }
        }
        tex->unbind();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_render_buf = RenderBufferData{ fbo, rbo, std::move(tex)};
    return {};
}

void EditorRenderer::clear_render_data() {
    std::vector<GLuint> vaos, vbos;
    for (auto&& [_, data] : m_render_data) {
        vaos.emplace_back(data.vao);
        vbos.emplace_back(data.vbo);
    }
    glDeleteBuffers(vbos.size(), vbos.data());
    glDeleteVertexArrays(vaos.size(), vaos.data());

    m_render_data.clear();
}
