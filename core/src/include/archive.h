#pragma once

#include <string_view>
#include <string>
#include <utils.h>
#include <tl/expected.hpp>

struct Scene;
struct Camera;

/**
 * \brief Interface for any serializer that can be used to save/load a scene.
*/
struct Archive {
    DEFAULT_COPY_MOVE(Archive);
    Archive() = default;
    virtual ~Archive() noexcept = default;
    
    /**
     * \brief Serializes the scene and camera to a string.
     * \param scene The scene to serialize.
     * \param camera The camera to serialize.
     * \return A string containing the serialized scene and camera. If an error occurs, an error message is returned.
    */
    virtual auto save(View<Scene> scene_view, View<Camera> camera_view) -> tl::expected<std::string, std::string> = 0;

    /**
     * \brief Deserializes the scene and camera from a string.
     * \return A pair containing the scene and camera. If an error occurs, an error message is returned.
    */
    virtual auto load(std::string_view data) -> tl::expected<std::pair<Scene, Camera>, std::string> = 0;
};