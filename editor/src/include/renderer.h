#pragma once
#include "commands.h"
#include "renderResult.h"
#include "scene.h"
#include "camera.h"
#include "renderConfig.h"

#include <tl/expected.hpp>


struct Renderer {
	explicit Renderer(RenderConfig const& config) noexcept;
    virtual ~Renderer() noexcept;

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
    [[nodiscard]] virtual auto open_scene(Scene scene) noexcept -> tl::expected<void, std::string> = 0;

	/**
     * \brief Executes a command.
     * \param cmd The command to be executed
     * \return on failure, a Result object that contains an error message\n
     * on success, an empty Result object.
     */
    [[nodiscard]] virtual auto exec(Cmd const& cmd) noexcept -> tl::expected<void, std::string> = 0;

	/**
     * \brief Renders the scene. Note that if you call this function outside of Application::loop(),\n
     * you will need to poll events, call glClear(), and swap buffers yourself.
     * \return on failure, a Result object that contains an error message\n
     * on success, a Result object that contains a handle to the rendered image.
     */
    [[nodiscard]] virtual auto render() noexcept -> tl::expected<void, std::string> = 0;

	/**
	 * \brief Renders the scene to an internal buffer. Note that if you call this function outside of Application::loop(),\n
	 * you will need to poll events, call glClear(), and swap buffers yourself.
	 * \return on failure, a Result object that contains an error message\n
	 * on success, a Result object that contains a handle to the rendered image.
	 */
    [[nodiscard]] virtual auto render_buffered() noexcept -> tl::expected<RenderResultRef, std::string> = 0;

    /**
     * \brief Checks if the renderer has a valid scene opened.
     * \return true if the renderer has a valid scene opened, false otherwise.
     */
    [[nodiscard]] virtual auto valid() const noexcept -> bool = 0;
    [[nodiscard]] auto get_config() const noexcept -> RenderConfig const& { return m_config; }

protected:
    [[nodiscard]] auto get_cam() noexcept -> Camera& { return m_cam; }

private:
    RenderConfig m_config;
    Camera m_cam;
};