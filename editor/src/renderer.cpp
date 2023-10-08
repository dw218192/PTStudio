#include "include/renderer.h"

#include <cassert>
#include <algorithm>

Renderer::Renderer(RenderConfig config) noexcept :
    m_config(config),
	m_cam{ config.fovy, m_config.width / static_cast<float>(m_config.height), Transform {} },
	m_vao(0)
{}

Renderer::~Renderer() noexcept {
    if(valid()) {
        assert(m_vao);
        for (GLuint const handle : m_buffer_handles) assert(handle);

        glDeleteBuffers(get_bufhandle_size(), get_bufhandles());
        glDeleteVertexArrays(1, &m_vao);
    }
}

auto Renderer::exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> {
    struct Handler {
        Handler (Renderer& rend) : rend(rend) { }
        void operator()(Cmd_CameraRot const& rot) const {
            rend.m_cam.set_rotation(TransformSpace::LOCAL, rot.angles_deg);
        }
        void operator()(Cmd_CameraMove const& rot) const {
            rend.m_cam.set_position(TransformSpace::LOCAL, rot.delta);
        }
        void operator()(Cmd_CameraZoom const& zoom) const {
            rend.m_cam.set_position(TransformSpace::LOCAL, { 0, 0, zoom.delta });
        }
        Renderer& rend;
    } handler{ *this };

    if(valid()) {
        std::visit(handler, cmd);
    }
    return {};
}

auto Renderer::open_scene(Scene scene) noexcept -> tl::expected<void, std::string> {
    GLenum err;
    auto roll_back = [this]() {
        if (m_vao) glDeleteVertexArrays(1, &m_vao);
        glDeleteBuffers(get_bufhandle_size(), get_bufhandles());
        m_vao = 0;
    };

#define CHECK_GL() do {\
	err = glGetError();\
	if(err != GL_NO_ERROR) {\
		roll_back();\
		return unexpected_gl_error(err);\
	} } while(false)

    if(!valid()) {
        // set up buffers
        glGenVertexArrays(1, &m_vao);

        m_buffer_handles.resize(1);
        glGenBuffers(1, m_buffer_handles.data());

        auto const res = m_res.init(m_config.width, m_config.height);
        if(!res) {
            return tl::unexpected{ res.error() };
        }
        CHECK_GL();
    }

    // fill buffers with data
    glBindVertexArray(m_vao);
    {
        glBindBuffer(GL_ARRAY_BUFFER, get_vbo());
        {
            // gather all vertices from all objects in the scene
            std::vector<Vertex> all_vertices;
            for(auto&& obj : scene.objects()) {
                all_vertices.insert(all_vertices.end(), obj.vertices().begin(), obj.vertices().end());
            }

            glBufferData(
                GL_ARRAY_BUFFER, 
                all_vertices.size() * sizeof(Vertex),
                all_vertices.data(),
                GL_STATIC_DRAW
            );
            CHECK_GL();
            
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void*>(offsetof(Vertex, position)));
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void*>(offsetof(Vertex, normal)));
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                reinterpret_cast<void*>(offsetof(Vertex, uv)));
            CHECK_GL();
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);
    CHECK_GL();

    m_scene = std::move(scene);

    // Set a few settings/modes in OpenGL rendering
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POLYGON_SMOOTH);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glBindVertexArray(m_vao);
    m_res.prepare();
    CHECK_GL();

    // Set up camera position so it has a good view of the object
    m_cam.set_transform(m_scene.get_good_cam_start());

    return {};

#undef CHECK_GL
}

auto Renderer::render() noexcept -> tl::expected<RenderResultRef, std::string> {
    if(!valid()) {
        return tl::unexpected { "render() called on invalid renderer" };
    }

    auto res = m_scene.begin_draw();
    if (!res) {
        return tl::unexpected{ res.error() };
    }

	for (size_t i = 0, vbo_offset = 0; i < m_scene.objects().size(); ++i) {
        auto&& obj = m_scene.objects()[i];

        res = obj.begin_draw(m_cam);
        if (!res) {
            return tl::unexpected{ res.error() };
        }
        glDrawArrays(GL_TRIANGLES,
            static_cast<GLint>(vbo_offset),
            static_cast<GLsizei>(obj.vertices().size()));
        vbo_offset += obj.vertices().size();

        obj.end_draw();
    }

    // check for errors
    auto const err = glGetError();
    if (err != GL_NO_ERROR) {
        return tl::unexpected{ reinterpret_cast<char const*>(glewGetErrorString(err)) };
    }

    return m_res;
}