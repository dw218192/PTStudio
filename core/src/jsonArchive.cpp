#include "jsonArchive.h"
#include "scene.h"
#include "camera.h"
#include "object.h"
#include "transform.h"
#include "boundingBox.h"


namespace {
	template<glm::length_t L, typename T>
	auto serialize(glm::vec<L, T> const& vec) -> nlohmann::json {
		nlohmann::json json;
		for (glm::length_t i = 0; i < L; ++i) {
			json.push_back(vec[i]);
		}
		return json;
	}
	template<glm::length_t L, typename T>
	auto serialize(glm::mat<L, L, T> const& mat) -> nlohmann::json {
		nlohmann::json json;
		for (glm::length_t i = 0; i < L; ++i) {
			json.push_back(serialize(mat[i]));
		}
		return json;
	}
	auto serialize(Transform const& transform) -> nlohmann::json {
		nlohmann::json json;
		json["position"] = serialize(transform.get_position());
		json["rotation"] = serialize(transform.get_rotation());
		json["scale"] = serialize(transform.get_scale());

		return json;
	}

	auto serialize(BoundingBox const& bound) -> nlohmann::json {
		nlohmann::json json;
		json["min"] = serialize(bound.min_pos);
		json["max"] = serialize(bound.max_pos);
		
		return json;
	}
	
	auto serialize(Material const& material) -> nlohmann::json {

	}

	auto serialize(Vertex const& vertex) -> nlohmann::json {
		nlohmann::json json;
		json["position"] = serialize(vertex.position);
		json["normal"] = serialize(vertex.normal);
		json["uv"] = serialize(vertex.uv);
		return json;
	}

	auto serialize(Object const& object) -> nlohmann::json {
		nlohmann::json json;
		json["bound"] = serialize(object.get_bound());
		json["transform"] = serialize(object.get_transform());
		json["material"] = serialize(object.get_material());
		auto vertices_obj = json["vertices"];
		for (auto vertex : object.get_vertices()) {
			vertices_obj.emplace_back(serialize(vertex));
		}
		json["name"] = object.get_name();
		
		return json;
	}
	template<glm::length_t L, typename T>
	auto deserialize(nlohmann::json const& json, glm::vec<L, T>& vec) -> tl::expected<void, std::string> {
		if (!json.is_array() || json.size() != L) {
			return TL_ERROR("expected array of length " + std::to_string(L));
		}
		for (glm::length_t i = 0; i < L; ++i) {
			vec[i] = json[i].get<T>();
		}
		return {};
	}

	template<glm::length_t L, typename T>
	auto deserialize(nlohmann::json const& json, glm::mat<L, L, T>& mat) -> tl::expected<void, std::string> {
		if (!json.is_array() || json.size() != L) {
			return TL_ERROR("expected array of length " + std::to_string(L));
		}
		for (glm::length_t i = 0; i < L; ++i) {
			TL_CHECK_AND_PASS(deserialize(json[i], mat[i]));
		}
		return {};
	}

	auto deserialize(nlohmann::json const& json, Transform& transform) -> tl::expected<void, std::string> {
		if (!json.is_object() || json.size() != 3) {
			return TL_ERROR("expected object");
		} else if (!json.contains("position") || !json.contains("rotation") || !json.contains("scale")) {
			return TL_ERROR("expected position, rotation, and scale");
		}

		glm::vec3 pos, rot, scale;
		TL_CHECK_AND_PASS(deserialize(json["position"], pos));
		TL_CHECK_AND_PASS(deserialize(json["rotation"], rot));
		TL_CHECK_AND_PASS(deserialize(json["scale"], scale));
		transform = Transform{ pos, rot, scale };
		return {};
	}
	auto deserialize(nlohmann::json const& json, Object& object) -> tl::expected<void, std::string> {
		object.set_name(json["name"].get<std::string>());
		Transform transform;
		deserialize(json["transform"], transform);
		object.set_transform(transform);
	}
}

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

auto JsonArchive::load(std::string_view data) -> tl::expected<std::pair<Scene, Camera>, std::string> {
}