#include "include/editorRenderer.h"
#include "include/application.h"

#include <cassert>
#include <algorithm>

constexpr auto k_grid_dim = 100.0f;
constexpr auto k_grid_spacing = 1.0f;
constexpr auto k_clear_color = glm::vec3{ 0 };
constexpr auto k_outline_scale = 1.02f;
constexpr auto k_outline_color = glm::vec3{ 1, 0, 0 };

EditorRenderer::EditorRenderer(RenderConfig const& config) noexcept
	: Renderer{config} { }

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

    // set up main frame buffer
    MainFrameBuffer::set(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	// set up shaders
    TL_ASSIGN(m_editor_shader, ShaderProgram::from_srcs(vs_obj_src, ps_obj_src));
    TL_ASSIGN(m_grid_shader, ShaderProgram::from_srcs(vs_grid_src, ps_grid_src));
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
        TL_ASSIGN(m_render_buf, GLFrameBuffer::create());
        TL_CHECK(m_render_buf->bind());
        {
	        TL_CHECK(m_render_buf->attach(m_config.width, m_config.height, {
		        {
					GL_COLOR_ATTACHMENT0, GL_RGB,
			        {
				        { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
				        { GL_TEXTURE_MAG_FILTER, GL_LINEAR },
			        }
		        },
		        {
		        	GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT,
		        	{
				        { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
				        { GL_TEXTURE_MAG_FILTER, GL_LINEAR },
		        	}
		        }
		    }));
            TL_CHECK(m_render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));
        }
        m_render_buf->unbind();
    }
    TL_CHECK(render_internal(cam, m_render_buf->handle()));
    return m_render_buf->get_texture(GL_COLOR_ATTACHMENT0);
}

auto EditorRenderer::on_change_render_config(RenderConfig const& config) noexcept -> tl::expected<void, std::string> {
    if (!config.is_valid() || config == m_config) {
        return {};
    }
    m_config = config;
    if (m_render_buf) {
        TL_CHECK_FWD(m_render_buf->bind());
        {
            TL_CHECK_FWD(m_render_buf->resize(m_config.width, m_config.height));
        }
        m_render_buf->unbind();
    }
    if (m_outline.render_buf) {
        TL_CHECK_FWD(m_outline.render_buf->bind());
        {
            TL_CHECK_FWD(m_outline.render_buf->resize(m_config.width, m_config.height));
        }
        m_outline.render_buf->unbind();
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
    auto const it = m_render_data.find(obj);
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
    auto draw_outline = [this, cam, fbo](ConstObjectHandle obj)-> tl::expected<void, std::string> {
        if (!m_outline.render_buf) {
            // initialize outline states

            // render buffer
            TL_ASSIGN(m_outline.render_buf, GLFrameBuffer::create());
            TL_CHECK_FWD(m_outline.render_buf->bind());
            {
                TL_CHECK_FWD(m_outline.render_buf->attach(m_config.width, m_config.height, {
                    {
                    	GL_COLOR_ATTACHMENT0, GL_RGBA,
                    	{
                            { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
							{ GL_TEXTURE_MAG_FILTER, GL_LINEAR },
                            { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE },
                    		{ GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE },
                    	}
                    },
                    {
                    	GL_COLOR_ATTACHMENT1, GL_RGBA,
                    	{
                            { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
                            { GL_TEXTURE_MAG_FILTER, GL_LINEAR },
                            { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE },
                            { GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE },
                        }
                    }
                }));
                TL_CHECK_FWD(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));
            }
            m_outline.render_buf->unbind();

            // shaders
            for (int i = 0; i < m_outline.shaders.size(); ++i) {
                TL_ASSIGN(m_outline.shaders[i], ShaderProgram::from_srcs(vs_outline_passes[i], ps_outline_passes[i]));
            }

            // full screen quad
            TL_ASSIGN(m_outline.quad_render_data, GLVertexArray::create(6));
            TL_CHECK_FWD(m_outline.quad_render_data->bind());
            {
                static constexpr float data[] = {
                    // First triangle (positions)
                    -1.0f, -1.0f, 0.0f,
                     1.0f, -1.0f, 0.0f,
                     1.0f,  1.0f, 0.0f,
                     // Second triangle (positions)
                     -1.0f, -1.0f, 0.0f,
                      1.0f,  1.0f, 0.0f,
                     -1.0f,  1.0f, 0.0f,
                     // First triangle (UVs)
                     0.0f, 0.0f,
                     1.0f, 0.0f,
                     1.0f, 1.0f,
                     // Second triangle (UVs)
                     0.0f, 0.0f,
                     1.0f, 1.0f,
                     0.0f, 1.0f
                };
                TL_CHECK_FWD(m_outline.quad_render_data->connect(tcb::make_span(data), 
                    GLAttributeInfo<glm::vec3>{0, 0, 0},
                    GLAttributeInfo<glm::vec2>{1, 0, 18 * sizeof(float)}
                ));
            }
            m_outline.quad_render_data->unbind();
        }

        TL_CHECK_FWD(m_outline.render_buf->bind());
        {
            TL_CHECK_FWD(m_outline.render_buf->clear(glm::vec3(0), 1.0f));

            // draw to attachment 0
            TL_CHECK(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));

            TL_CHECK_FWD(m_outline.shaders[0]->bind());
            {
                TL_CHECK_FWD(m_outline.shaders[0]->set_uniform(k_uniform_view, cam.get_view()));
                TL_CHECK_FWD(m_outline.shaders[0]->set_uniform(k_uniform_projection, cam.get_projection()));
                TL_CHECK_FWD(m_outline.shaders[0]->set_uniform(k_uniform_model, obj->get_transform().get_matrix()));

                auto&& vao = m_render_data.at(obj);
                TL_CHECK_FWD(vao->bind());
                TL_CHECK_FWD(vao->draw_array(GL_TRIANGLES));
            }
            // draw to attachment 1 based on attachment 0
            TL_CHECK(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT1));
            TL_CHECK_FWD(m_outline.shaders[1]->bind());
            {
                TL_CHECK_FWD(m_outline.shaders[1]->set_texture(k_uniform_screen_texture, m_outline.render_buf->get_texture(GL_COLOR_ATTACHMENT0), 0));
                TL_CHECK_FWD(m_outline.shaders[1]->set_uniform(k_uniform_outline_color, glm::vec3(1, 0, 0)));
                TL_CHECK_FWD(m_outline.shaders[1]->set_uniform(k_uniform_thickness, 0.9f));
                TL_CHECK_FWD(m_outline.shaders[1]->set_uniform(k_uniform_texel_size, 1.0f / glm::vec2(m_config.width, m_config.height)));
                TL_CHECK_FWD(m_outline.quad_render_data->bind());
                TL_CHECK_FWD(m_outline.quad_render_data->draw_array(GL_TRIANGLES));
                m_outline.quad_render_data->unbind();
            }
        }
        m_outline.render_buf->unbind();

        // attachment 1 now contains the outline
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (!fbo) {
            TL_CHECK_FWD(MainFrameBuffer::bind());
        } else {
            TL_CHECK_FWD(m_render_buf->bind());
        }

        TL_CHECK_FWD(m_outline.shaders[2]->bind());
        {
            TL_CHECK_FWD(m_outline.quad_render_data->bind());
            {
                TL_CHECK_FWD(m_outline.shaders[2]->set_texture(k_uniform_screen_texture, m_outline.render_buf->get_texture(GL_COLOR_ATTACHMENT1), 0));
                TL_CHECK_FWD(m_outline.quad_render_data->draw_array(GL_TRIANGLES));
            }
            m_outline.quad_render_data->unbind();
        }
        m_outline.shaders[2]->unbind();

        glDisable(GL_BLEND);

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
    if (!fbo) {
        TL_CHECK_FWD(MainFrameBuffer::bind());
        TL_CHECK_FWD(MainFrameBuffer::clear(k_clear_color, 1.0f));
    } else {
        TL_CHECK_FWD(m_render_buf->bind());
        TL_CHECK_FWD(m_render_buf->clear(k_clear_color, 1.0f));
    }

    glViewport(0, 0, static_cast<GLsizei>(get_config().width), static_cast<GLsizei>(get_config().height));

    // render objects
    TL_CHECK_FWD(m_editor_shader->bind());
    {
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_light_pos, scene.get_good_light_pos()));
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_view, cam.get_view()));
        TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_projection, cam.get_projection()));

        for (auto obj : scene) {
            TL_CHECK_FWD(m_editor_shader->set_uniform(k_uniform_model, obj->get_transform().get_matrix()));
            auto const it = m_render_data.find(obj);
            if (it == m_render_data.end()) {
                return  TL_ERROR("obj not found in render data");
            }
            auto&& vao = it->second;
            TL_CHECK_FWD(vao->bind());
            TL_CHECK_FWD(vao->draw_array(GL_TRIANGLES));
        }
    }
    m_editor_shader->unbind();



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

    // render outline
    if (m_cur_outline_obj) {
        TL_CHECK_FWD(draw_outline(m_cur_outline_obj));
    }

    // reset state
    if (fbo) {
        TL_CHECK_FWD(MainFrameBuffer::bind());
    }
    return {};
}

void EditorRenderer::clear_render_data() {
    m_render_data.clear();
}
