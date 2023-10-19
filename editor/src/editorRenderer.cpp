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

EditorRenderer::~EditorRenderer() noexcept { }

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

        TL_ASSIGN(m_grid_render_data, GLVertexArray::create(vertices.size()));
        TL_CHECK_FWD(m_grid_render_data->bind());
    	{
            auto const& view = vertices;
            TL_CHECK_FWD(m_grid_render_data->connect(tcb::make_span(view), GLAttributeInfo<glm::vec3>{0, 0, 0}));
        }
        m_grid_render_data->unbind();

        CHECK_GL_ERROR();

    	// set invariant uniforms
        TL_CHECK_FWD(m_grid_shader->bind());
        {
            TL_CHECK_FWD(m_grid_shader->set_uniform(k_uniform_half_grid_dim, half_dim));
        }
        m_grid_shader->unbind();

        return {};
    };

	if (valid()) {
        return {};
    }

	// set up shaders
    TL_ASSIGN(m_editor_shader, ShaderProgram::from_srcs(vs_obj_src, ps_obj_src));
    TL_ASSIGN(m_grid_shader, ShaderProgram::from_srcs(vs_grid_src, ps_grid_src));
    TL_ASSIGN(m_outline_shader, ShaderProgram::from_srcs(vs_outline_src, ps_outline_src));
    TL_CHECK_FWD(m_outline_shader->bind());
    {
        TL_CHECK_FWD(m_outline_shader->set_uniform(k_uniform_outline_color, k_outline_color));
    }
    m_outline_shader->unbind();

    TL_CHECK_FWD(create_grid(k_grid_dim, k_grid_spacing));

    m_valid = true;
    return {};
}

auto EditorRenderer::open_scene(Scene const& scene) noexcept -> tl::expected<void, std::string> {
    m_cur_outline_obj = nullptr;
    clear_render_data();

    CHECK_GL_ERROR();

    for (auto const obj : scene) {
        TL_CHECK_FWD(on_add_object(obj));
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
        TL_CHECK(create_render_buf());
    }
    TL_CHECK(render_internal(cam, m_render_buf->fbo->handle()));
    return m_render_buf->tex.get();
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
    } else if (m_render_data.count(obj)) {
        return TL_ERROR("object added twice");
    }
    TL_CHECK_FWD(on_add_object_internal(m_render_data[obj], obj));

    return {};
}

auto EditorRenderer::on_remove_object(ConstObjectHandle obj) noexcept -> tl::expected<void, std::string> {
    auto it = m_render_data.find(obj);
    if(it == m_render_data.end()) {
        return TL_ERROR("on_remove_object: obj not found");
    }
    m_render_data.erase(obj);

    return {};
}

void EditorRenderer::on_object_change(ConstObjectHandle obj) noexcept {
    m_cur_outline_obj = obj;
}

auto EditorRenderer::on_add_object_internal(GLVertexArrayRef& data, ConstObjectHandle obj) noexcept -> tl::expected<void, std::string> {
    TL_ASSIGN(data, GLVertexArray::create(obj->get_vertices().size()));
    TL_CHECK_FWD(data->bind());
    {
        TL_CHECK_FWD(data->connect(obj->get_vertices(),
            GLAttributeInfo<glm::vec3> { 0, sizeof(Vertex), offsetof(Vertex, position) },
            GLAttributeInfo<glm::vec3> { 1, sizeof(Vertex), offsetof(Vertex, normal) },
            GLAttributeInfo<glm::vec3> { 2, sizeof(Vertex), offsetof(Vertex, uv) }
        ));
    }
    data->unbind();

    return {};

}

auto EditorRenderer::render_internal(Camera const& cam, GLuint fbo) noexcept -> tl::expected<void, std::string> {
    auto draw_obj = [this](ConstObjectHandle obj) -> tl::expected<void, std::string> {
        auto const it = m_render_data.find(obj);
        if (it == m_render_data.end()) {
            return  TL_ERROR("obj not found in render data");
        }
        auto&& [_, vao] = *it;
        TL_CHECK_FWD(vao->bind());
        TL_CHECK_FWD(vao->draw_array(GL_TRIANGLES));
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
    TL_CHECK_FWD(m_editor_shader->bind());
    {
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_light_pos, scene.get_good_light_pos()));
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_view, cam.get_view()));
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_projection, cam.get_projection()));

        for (auto obj : scene) {
            TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_model, obj->get_transform().get_matrix()));
            TL_CHECK_FWD(draw_obj(obj));
        }
    }
    m_editor_shader->unbind();

    // draw outlined obj if some obj is selected
    // this renders that obj 3 times which is bad
    // TODO: optimize if too slow?
    if (m_cur_outline_obj) {
        TL_CHECK_FWD(m_outline_shader->bind());
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
        m_outline_shader->unbind();
    }

    // render grid
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    TL_CHECK_FWD(m_grid_render_data->bind());
    TL_CHECK_FWD(m_grid_shader->bind());
    {
        TL_CHECK_FWD(m_grid_shader->set_uniform(k_uniform_view, cam.get_view()));
        TL_CHECK_FWD(m_grid_shader->set_uniform(k_uniform_projection, cam.get_projection()));
        TL_CHECK_FWD(m_grid_render_data->draw_array(GL_LINES));
    }
    m_grid_shader->unbind();
    m_grid_render_data->unbind();

    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return {};
}

auto EditorRenderer::create_render_buf() noexcept -> tl::expected<void, std::string> {
    GLTextureRef tex;
    GLFrameBufferRef fbo;
    GLRenderBufferRef rbo;

    auto const w = static_cast<GLsizei>(get_config().width);
    auto const h = static_cast<GLsizei>(get_config().height);

    if (m_render_buf) {
        // temporarily move ownership
        fbo = std::move(m_render_buf->fbo);
        tex = std::move(m_render_buf->tex);
        rbo = std::move(m_render_buf->rbo);

        TL_CHECK_FWD(tex->resize(w, h));
    } else {
        // create tex to render to
        TL_ASSIGN(tex, GLTexture::create(w, h));
        TL_ASSIGN(fbo, GLFrameBuffer::create());
        TL_ASSIGN(rbo, GLRenderBuffer::create(w, h, GL_DEPTH24_STENCIL8_OES));
    }
    
    TL_CHECK_FWD(fbo->bind());
    {
        TL_CHECK_FWD(rbo->bind());
        {
            TL_CHECK_FWD(fbo->attach(GL_DEPTH_ATTACHMENT, rbo.get()));
            TL_CHECK_FWD(fbo->attach(GL_STENCIL_ATTACHMENT, rbo.get()));
        }
        rbo->unbind();

        CHECK_GL_ERROR();

        TL_CHECK_FWD(tex->bind());
        {
            TL_CHECK_FWD(fbo->attach(GL_COLOR_ATTACHMENT0, tex.get()));
            TL_CHECK_FWD(fbo->set_draw_buffer(GL_COLOR_ATTACHMENT0));

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                return TL_ERROR(std::string{ "frame buffer is not valid, status = " } +
                    std::to_string(glCheckFramebufferStatus(GL_FRAMEBUFFER)));
            }
        }
        tex->unbind();
    }
    fbo->unbind();

    m_render_buf = RenderBufferData{ std::move(fbo), std::move(rbo), std::move(tex)};
    return {};
}

void EditorRenderer::clear_render_data() {
    m_render_data.clear();
}
