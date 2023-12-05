#include "sceneObject.h"
#include "camera.h"
#include "scene.h"

#include <tiny_obj_loader.h>
#include <glm/ext/scalar_constants.hpp>

PTS::SceneObject::SceneObject(ObjectConstructorUsage usage) noexcept
    : Object{ usage } {}

PTS::SceneObject::SceneObject(Scene const& scene, std::string_view name, Transform transform)
    : Object{ name }, m_transform{ std::move(transform) }, m_scene{ &scene } {}

PTS::SceneObject::SceneObject(Scene const& scene, Transform transform)
    : Object{ scene.next_obj_name() }, m_transform{ std::move(transform) }, m_scene{ &scene } {}
