#pragma once
#include "texture.h"
#include "scene.h"
#include "camera.h"
#include "renderConfig.h"
#include "utils.h"

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

    NODISCARD virtual auto init() noexcept -> tl::expected<void, std::string> = 0;

	/**
	 * \brief Opens a new scene
	 * \param scene The scene to be opened
	 * \return on failure, an error message
	 */
    NODISCARD virtual auto open_scene(Scene const& scene) noexcept -> tl::expected<void, std::string> = 0;

    /**
     * \brief Changes the render configuration
     * \param config The new render configuration
     * \return on failure, an error message
    */
    NODISCARD virtual auto on_change_render_config(RenderConfig const& config) noexcept -> tl::expected<void, std::string> = 0;

    /**
     * \brief Called when an object is added to the scene
     * \param obj The object that was added
     * \return on failure, an error message
    */
    NODISCARD virtual auto on_add_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> = 0;

    /**
     * \brief Called when an object is removed from the scene
     * \param obj The object that was removed
     * \return on failure, an error message
    */
    NODISCARD virtual auto on_remove_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> = 0;

	/**
     * \brief Renders the scene directly in the window
     * \param camera The camera to view the scene from
     * \return on failure, an error message
     */
    NODISCARD virtual auto render(Camera const& camera) noexcept -> tl::expected<void, std::string> = 0;

	/**
	 * \brief Renders the scene to a texture
     * \param camera The camera to view the scene from
     * \return on failure, an error message\n
     * on success, a handle to the texture containing the rendered scene
     * \note The texture is owned by the renderer and will be deleted when the renderer is destroyed
     */
    NODISCARD virtual auto render_buffered(Camera const& camera) noexcept -> tl::expected<TextureHandle, std::string> = 0;

    /**
     * \brief Checks if the renderer is initialized and valid.
     * \return true if the renderer is initialized and valid, false otherwise.
     */
    NODISCARD virtual auto valid() const noexcept -> bool = 0;

    /**
     * \brief Gets the current render configuration
     * \return The current render configuration
     */
    NODISCARD auto get_config() const noexcept -> RenderConfig const& { return m_config; }

    /**
	 * \brief Draws any custom UI that might help editing that is specific to a renderer
	 * \return on failure, an error message
	 */
    NODISCARD virtual auto draw_imgui() noexcept -> tl::expected<void, std::string> {
        return {};
    }
protected:
    RenderConfig m_config;
};