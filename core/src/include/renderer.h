#pragma once
#include "texture.h"
#include "scene.h"
#include "camera.h"
#include "renderConfig.h"
#include "utils.h"

#include <tl/expected.hpp>

enum class EditableChangeType {
    TRANSFORM,
    MATERIAL,
    EDIT_FLAGS,
    OTHER
};

namespace PTS {
    struct Application;
    struct Renderer {
        NO_COPY_MOVE(Renderer);

        explicit Renderer(RenderConfig config, std::string_view name) noexcept;
        virtual ~Renderer() noexcept;

        NODISCARD virtual auto init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string>;
        /**
         * \brief Opens a new scene and closes the current one if there is one
         * \param scene The scene to be opened
         * \return on failure, an error message
         */
        NODISCARD virtual auto open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> = 0;

        /**
         * \brief Changes the render configuration
         * \param config The new render configuration
         * \return on failure, an error message
        */
        NODISCARD auto set_render_config(RenderConfig config) noexcept -> tl::expected<void, std::string>;

        /**
         * \brief Called when an object is added to the scene
         * \param editable The editable that was added
         * \return on failure, an error message
        */
        NODISCARD virtual auto on_add_editable(EditableView editable) noexcept -> tl::expected<void, std::string> = 0;

        /**
         * \brief Called when an object is removed from the scene
         * \param editable The editable that was removed
         * \return on failure, an error message
        */
        NODISCARD virtual auto on_remove_editable(EditableView editable) noexcept -> tl::expected<void, std::string> = 0;

        /**
         * \brief Called when an editable is changed in any way (transform, material, etc.)
         * \param editable The editable that was changed
         * \param type The type of change that was made
         * \return on failure, an error message
        */
        NODISCARD virtual auto on_editable_change(EditableView editable, EditableChangeType type) noexcept -> tl::expected<void, std::string> = 0;

        /**
         * \brief Renders the scene directly in the window
         * \param camera The camera to view the scene from
         * \return on failure, an error message
         */
        NODISCARD virtual auto render(View<Camera> camera) noexcept -> tl::expected<void, std::string> = 0;

        /**
         * \brief Renders the scene to a texture
         * \param camera The camera to view the scene from
         * \return on failure, an error message\n
         * on success, a handle to the texture containing the rendered scene
         * \note The texture is owned by the renderer and will be deleted when the renderer is destroyed
         */
        NODISCARD virtual auto render_buffered(View<Camera> camera) noexcept -> tl::expected<TextureHandle, std::string> = 0;

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
         * \brief Gets the name of the renderer
         * \return The name of the renderer
         */
        NODISCARD auto get_name() const noexcept -> std::string_view { return m_name; }

        /**
         * \brief Draws any custom UI that might help editing that is specific to a renderer
         * \return on failure, an error message
         */
        NODISCARD virtual auto draw_imgui() noexcept -> tl::expected<void, std::string> {
            return {};
        }

    protected:
        NODISCARD virtual auto on_change_render_config() noexcept -> tl::expected<void, std::string> = 0;

        std::string m_name;
        RenderConfig m_config;
        ObserverPtr<Application> m_app;
    };
}