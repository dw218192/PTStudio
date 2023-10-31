#include "jsonArchive.h"
#include "scene.h"
#include "camera.h"

auto JsonArchive::save(View<Scene> scene_view, View<Camera> camera_view) -> tl::expected<std::string, std::string> {
	auto&& scene = scene_view.get();
	auto&& camera = camera_view.get();

	auto&& scene_obj = m_json[scene.get_name()];
	for (auto object : scene) {
		if (!object) {
			return TL_ERROR("Scene contains null object");
		}
		scene_obj.emplace_back(serialize(*object));
	}

	m_json["camera"] = {
		camera.get_fov(),
		camera.get_px_width(),
		camera.get_px_height(),
		serialize(camera.get_eye()),
		serialize(camera.get_center()),
		serialize(camera.get_up())
	};
}

auto JsonArchive::load() -> tl::expected<std::pair<Scene, Camera>, std::string> {
}

auto JsonArchive::serialize(Transform const& transform) -> nlohmann::json {
	nlohmann::json json;
	json["position"] = serialize(transform.get_position());
	json["rotation"] = serialize(transform.get_rotation());
	json["scale"] = serialize(transform.get_scale());
	return json;
}

auto JsonArchive::serialize(Object const& object) -> nlohmann::json {
	nlohmann::json json;
	json["name"] = object.get_name();
	json["transform"] = serialize(object.get_transform());
	return json;
}
