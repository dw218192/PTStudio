#include "include/editorRenderer.h"

#include <cassert>
#include <algorithm>

constexpr char const* ps_unicolor_src =
"\
	#version 330 core\n\
	layout(location = 0) out vec3 FragColor; \n\
	void main() {\n\
        FragColor = vec3(1.0f, 0.5f, 0.2f); \n\
	}\n\
";

constexpr char const* vs_obj_src =
"\
    #version 330 core\n\
    layout (location = 0) in vec3 aPos;\n\
    layout (location = 1) in vec3 aNormal;\n\
    layout (location = 2) in vec2 aTexCoords;\n\
    out vec2 TexCoords;\n\
    out vec3 Normal;\n\
    out vec3 FragPos;\n\
    uniform mat4 model;\n\
    uniform mat4 view;\n\
    uniform mat4 projection;\n\
    void main() {\n\
        TexCoords = aTexCoords;\n\
        Normal = mat3(transpose(inverse(model))) * aNormal;\n\
        FragPos = vec3(model * vec4(aPos, 1.0));\n\
        gl_Position = projection * view * vec4(FragPos, 1.0);\n\
    }\n\
";

constexpr char const* ps_obj_src =
"\
    #version 330 core\n\
    out vec4 FragColor;\n\
    in vec2 TexCoords;\n\
    in vec3 Normal;\n\
    in vec3 FragPos;\n\
	uniform vec3 lightPos;\n\
    uniform mat4 view;\n\
    void main() {\n\
        const vec3 objectColor = vec3(178.0/255.0, 190.0/255.0, 181.0/255.0);\n\
        const vec3 lightColor = vec3(1.0, 1.0, 1.0);\n\
		vec3 camPos = view[3].xyz;\n\
        float ambientStrength = 0.2;\n\
        vec3 ambient = ambientStrength * lightColor;\n\
        vec3 norm = normalize(Normal);\n\
        vec3 lightDir = normalize(lightPos - FragPos);\n\
        float diff = max(dot(norm, lightDir), 0.0);\n\
        vec3 diffuse = diff * lightColor;\n\
        float specularStrength = 0.5;\n\
        vec3 viewDir = normalize(camPos - FragPos);\n\
        vec3 reflectDir = reflect(-lightDir, norm);\n\
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);\n\
        vec3 specular = specularStrength * spec * lightColor;\n\
        vec3 result = (ambient + diffuse + specular) * objectColor;\n\
        FragColor = vec4(result, 1.0);\n\
    }\n\
";

constexpr char const* k_uniform_model = "model";
constexpr char const* k_uniform_view = "view";
constexpr char const* k_uniform_projection = "projection";
constexpr char const* k_uniform_light_pos = "lightPos";
constexpr char const* k_uniform_light_color = "lightColor";
constexpr char const* k_uniform_object_color = "objectColor";


EditorRenderer::EditorRenderer(RenderConfig const& config) noexcept
	: Renderer{config}, m_res{config.width, config.height} {}

EditorRenderer::~EditorRenderer() {
    if (EditorRenderer::valid()) {
        glDeleteBuffers(get_bufhandle_size(), get_bufhandles());
        glDeleteVertexArrays(1, &m_vao);
        glDeleteFramebuffers(1, &m_rend_buf.fbo);
        glDeleteRenderbuffers(1, &m_rend_buf.depth_rbo);
        glDeleteTextures(1, &m_rend_buf.tex);
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
        void operator()(Cmd_ChangeRenderConfig const& new_config) const {
            
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
        // set up object buffers
        glGenVertexArrays(1, &m_vao);
        m_buffer_handles.resize(1);
        glGenBuffers(1, m_buffer_handles.data());

        err = glGetError();
        if (err != GL_NO_ERROR) {
            return unexpected_gl_error(err);
        }

        // set up render buffer
        auto res = create_frame_buf();
        if (!res) {
            return tl::unexpected{ res.error() };
        } else {
            m_rend_buf = std::move(res.value());
        }

        // set up shader
        auto shader_res = create_default_shader();
        if (!shader_res) {
            return tl::unexpected{ shader_res.error() };
        } else {
            m_editor_shader = std::move(shader_res.value());
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
                all_vertices.insert(all_vertices.end(), obj.get_vertices().begin(), obj.get_vertices().end());
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
    auto res = render_internal(m_rend_buf.fbo);
    if (!res) {
        return tl::unexpected{ res.error() };
    }

    auto& pixels = m_res.get_pixels();
    auto w = m_res.get_width(), h = m_res.get_height();

    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    // flip vertically
    for (decltype(w) y = 0; y < h / 2; ++y) {
        for (decltype(w) x = 0; x < w; ++x) {
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

    m_editor_shader.use();
    m_editor_shader.set_uniform(k_uniform_light_pos, m_scene.get_good_light_pos());
    auto res = m_editor_shader.set_uniform(k_uniform_view, get_cam().get_transform().get_matrix());
    if (!res) {
        return res;
    }
    res = m_editor_shader.set_uniform(k_uniform_projection, get_cam().get_projection());
    if (!res) {
        return res;
    }

	for (size_t i = 0, vbo_offset = 0; i < m_scene.objects().size(); ++i) {
        auto&& obj = m_scene.objects()[i];

        res = m_editor_shader.set_uniform(k_uniform_model, obj.get_transform().get_matrix());
        if (!res) {
            return res;
        }

        glDrawArrays(GL_TRIANGLES,
            static_cast<GLint>(vbo_offset),
            static_cast<GLsizei>(obj.get_vertices().size()));
        vbo_offset += obj.get_vertices().size();
    }

    return {};
}

auto EditorRenderer::create_default_shader() noexcept -> tl::expected<ShaderProgram, std::string> {
    auto vs_res = Shader::from_src(ShaderType::Vertex, vs_obj_src);
    if (!vs_res) {
        return tl::unexpected{ vs_res.error() };
    }
    auto ps_res = Shader::from_src(ShaderType::Fragment, ps_obj_src);
    if (!ps_res) {
        return tl::unexpected{ ps_res.error() };
    }

    return ShaderProgram::from_shaders(std::move(vs_res.value()), std::move(ps_res.value()));
}

auto EditorRenderer::create_frame_buf() noexcept -> tl::expected<RenderBuf, std::string> {
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

    return RenderBuf { fbo, rbo, tex };
}
