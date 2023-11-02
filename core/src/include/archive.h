#pragma once

#include <string_view>
#include <string>
#include <utils.h>
#include <tl/expected.hpp>

#include "scene.h"
#include "camera.h"

/**
 * \brief Interface for any serializer that can be used to save/load a scene.
*/
struct Archive {
    DEFAULT_COPY_MOVE(Archive);
    Archive() = default;
    virtual ~Archive() noexcept = default;

    virtual auto get_ext() -> std::string_view = 0;

    /**
     * \brief Serializes the scene and camera to a string.
     * \param scene_view The scene to serialize.
     * \param camera_view The camera to serialize.
     * \return A string containing the serialized scene and camera. If an error occurs, an error message is returned.
    */
    virtual auto save(View<Scene> scene_view, View<Camera> camera_view) -> tl::expected<std::string, std::string> = 0;

    /**
     * \brief Deserializes the scene and camera from a string.
     * \return A pair containing the scene and camera. If an error occurs, an error message is returned.
    */
    virtual auto load(std::string_view data) -> tl::expected<std::pair<Scene, Camera>, std::string> = 0;

    auto load_file(std::string_view file) noexcept -> tl::expected<std::pair<Scene, Camera>, std::string>;
    auto save_file(View<Scene> scene_view, View<Camera> camera_view, std::string_view file) noexcept -> tl::expected<void, std::string>;
};