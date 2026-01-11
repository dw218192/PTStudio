#pragma once
#include <tl/expected.hpp>

#include "camera.h"
#include "renderConfig.h"
#include "scene.h"
#include "sceneObject.h"
#include "texture.h"
#include "utils.h"

namespace PTS {
struct Application;

struct Renderer {
    NO_COPY_MOVE(Renderer);

    explicit Renderer(RenderConfig config, std::string_view name) noexcept;
    virtual ~Renderer() noexcept;

    NODISCARD virtual auto init(ObserverPtr<Application> app) noexcept
        -> tl::expected<void, std::string>;
    /**
     * \brief Opens a new scene and closes the current one if there is one
     * \param scene The scene to be opened
     * \return on failure, an error message
     */
    NODISCARD virtual auto open_scene(Ref<Scene> scene) noexcept
        -> tl::expected<void, std::string> = 0;

    /**
     * \brief Changes the render configuration
     * \param config The new render configuration
     * \return on failure, an error message
     */
    NODISCARD auto set_render_config(RenderConfig config) noexcept
        -> tl::expected<void, std::string>;

    /**
     * \brief Renders the scene to a texture
     * \param camera The camera to view the scene from
     * \return on failure, an error message\n
     * on success, a handle to the texture containing the rendered scene
     * \note The texture is owned by the renderer and will be deleted when the renderer is destroyed
     */
    NODISCARD virtual auto render(View<Camera> camera) noexcept
        -> tl::expected<TextureHandle, std::string> = 0;

    /**
     * \brief Checks if the renderer is initialized and valid.
     * \return true if the renderer is initialized and valid, false otherwise.
     */
    NODISCARD virtual auto valid() const noexcept -> bool = 0;

    /**
     * \brief Gets the current render configuration
     * \return The current render configuration
     */
    NODISCARD auto get_config() const noexcept -> RenderConfig const& {
        return m_config;
    }

    /**
     * \brief Gets the name of the renderer
     * \return The name of the renderer
     */
    NODISCARD auto get_name() const noexcept -> std::string_view {
        return m_name;
    }

    /**
     * \brief Draws any custom UI that might help editing that is specific to a renderer
     * \return on failure, an error message
     */
    NODISCARD virtual auto draw_imgui() noexcept -> tl::expected<void, std::string> {
        return {};
    }

   protected:
    NODISCARD virtual auto on_change_render_config() noexcept
        -> tl::expected<void, std::string> = 0;

    std::string m_name;
    RenderConfig m_config;
    ObserverPtr<Application> m_app;
};
}  // namespace PTS
