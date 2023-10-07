#pragma once
#include "result.h"
#include "commands.h"
#include "renderResult.h"
#include "scene.h"
#include "camera.h"
#include <vector>

struct RenderConfig {
	RenderConfig(unsigned width, unsigned height, float fovy, float max_fps)
		: width{width}, height{height}, fovy{fovy}, max_fps{max_fps}, min_frame_time{1.0f / max_fps}
	{ }

	unsigned width, height;
    float fovy;
    float max_fps;
    float min_frame_time;
};

/**
 * \brief This class is responsible for rendering a scene and managing the underlying OpenGL resources.\n
 * Normally you would not need to create this class directly.\n
 * Instead, the Application class will create & manage it for you.
 */
struct Renderer {
    /**
     * \brief Constructs an empty renderer\n
     * use open_scene() to open a scene and fully initialize the renderer.
     * \param config The renderer configuration to be used
     */
	explicit Renderer(RenderConfig config) noexcept;
    ~Renderer() noexcept;

	// don't copy because we have handles to GL resources
	Renderer(Renderer&) = delete;
    Renderer& operator=(Renderer&) = delete;

    // may implement move, but delete them for now
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

	/**
	 * \brief Opens a scene and initializes the renderer
	 * \param scene The scene to be opened
	 * \return on failure, a Result object that contains an error message\n
	 * on success, an empty Result object.
	 */
	[[nodiscard]] auto open_scene(Scene scene) noexcept -> Result<void>;
    /**
     * \brief Executes a command.
     * \param cmd The command to be executed
     * \return on failure, a Result object that contains an error message\n
     * on success, an empty Result object.
     */
    [[nodiscard]] auto exec(Cmd const& cmd) noexcept -> Result<void>;
    /**
     * \brief Renders the scene. Note that if you call this function outside of Application::loop(),\n
     * you will need to poll events, call glClear(), and swap buffers yourself.
     * \return on failure, a Result object that contains an error message\n
     * on success, a Result object that contains a handle to the rendered image.
     */
    [[nodiscard]] auto render() noexcept -> Result<RenderResult const&>;

    /**
     * \brief Checks if the renderer has a valid scene opened.
     * \return true if the renderer has a valid scene opened, false otherwise.
     */
    [[nodiscard]] auto valid() const noexcept -> bool { return m_vao != 0; }
    [[nodiscard]] auto get_config() const noexcept -> RenderConfig const& { return m_config; }
private:
    RenderConfig m_config;
    Camera m_cam;
    RenderResult m_res;
    Scene m_scene;

    // TODO: may abstract these into a class later
    // use 1 vao, 1 vbo for all meshes (objects)
	GLuint m_vao;

    [[nodiscard]] auto get_vbo() const noexcept { return m_buffer_handles[0]; }
    [[nodiscard]] auto get_bufhandle_size() const noexcept { return static_cast<GLsizei>(m_buffer_handles.size()); }
    [[nodiscard]] auto get_bufhandles() const noexcept { return m_buffer_handles.data(); }
    std::vector<GLuint> m_buffer_handles;
};