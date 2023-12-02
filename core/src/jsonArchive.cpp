#include "jsonArchive.h"
#include "scene.h"
#include "camera.h"
#include "object.h"
#include "transform.h"
#include "utils.h"

namespace glm {
	template<length_t L, typename T, qualifier Q>
	auto to_json(nlohmann::json& json, vec<L, T, Q> const& vec) -> void {
		for (length_t i = 0; i < L; ++i) {
			json.push_back(vec[i]);
		}
	}
	template<length_t L, typename T, qualifier Q>
	auto to_json(nlohmann::json& json, mat<L, L, T, Q> const& mat) -> void {
		for (length_t i = 0; i < L; ++i) {
			json.push_back(mat[i]);
		}
	}
	template<length_t L, typename T>
	auto from_json(nlohmann::json const& json, vec<L, T>& vec) -> void {
		for (length_t i = 0; i < L; ++i) {
			json.at(i).get_to(vec[i]);
		}
	}
	template<length_t L, typename T>
	auto from_json(nlohmann::json const& json, mat<L, L, T>& mat) -> void {
		for (length_t i = 0; i < L; ++i) {
			json.at(i).get_to(mat[i]);
		}
	}
}

namespace PTS {
	template<typename Reflected, typename = std::enable_if_t<is_reflectable<Reflected>::value>>
	auto to_json(nlohmann::json& json, ObserverPtr<Reflected> ptr) -> void {
		to_json(json, ViewPtr<Reflected> { ptr });
	}
	template<typename Reflected, typename = std::enable_if_t<is_reflectable<Reflected>::value>>
	auto from_json(nlohmann::json const& json, ObserverPtr<Reflected>& ptr) -> void {
		from_json(json, ViewPtr<Reflected> { ptr });
	}
	template<typename Reflected, typename = std::enable_if_t<is_reflectable<Reflected>::value>>
	auto to_json(nlohmann::json& json, ViewPtr<Reflected> ptr) -> void {
		// TODO
	}
	template<typename Reflected, typename = std::enable_if_t<is_reflectable<Reflected>::value>>
	auto from_json(nlohmann::json const& json, ViewPtr<Reflected>& ptr) -> void {
		// TODO
	}
	template<typename Reflected, typename = std::enable_if_t<is_reflectable<Reflected>::value>>
	auto to_json(nlohmann::json& json, Reflected const& reflected) -> void {
		if constexpr (has_serialization_callback<Reflected>::value) {
			reflected.on_serialize();
		}
		Reflected::for_each_field([&reflected, &json](auto field) {
			if (field.template get_modifier<MSerialize>()) {
				json[field.var_name] = field.get(reflected);
			}
		});
	}
	
	// for compatibility:
	// 1. if a new field is present in the json but not in the reflected type, it will be ignored
	// 2. if a new field is present in the reflected type but not in the json, it will be initialized with the default value
	template<typename Reflected, typename = std::enable_if_t<is_reflectable<Reflected>::value>>
	auto from_json(nlohmann::json const& json, Reflected& reflected) -> void {
		Reflected::for_each_field([&reflected, &json](auto field) {
			if (field.template get_modifier<MSerialize>()) {
				if (json.count(field.var_name)) {
					from_json(json.at(field.var_name), field.get(reflected));
				} else {
					// use default value if not present in json
					field.get(reflected) = field.get_default();
				}
			}
		});
		if constexpr (has_deserialization_callback<Reflected>::value) {
			reflected.on_deserialize();
		}
	}

	auto JsonArchive::save(View<Scene> scene_view, View<Camera> camera_view) -> tl::expected<std::string, std::string> {
		auto&& scene = scene_view.get();
		auto&& camera = camera_view.get();
		nlohmann::json json;

		try {
			to_json(json["scene"], scene);
			to_json(json["camera"], camera);
		}
		catch (nlohmann::json::exception const& e) {
			return TL_ERROR("Failed to serialize: " + std::string{ e.what() });
		}

		return json.dump();
	}

	auto JsonArchive::load(std::string_view data) -> tl::expected<std::pair<Scene, Camera>, std::string> {
		auto json = nlohmann::json::parse(data);
		try {
			Scene scene;
			from_json(json.at("scene"), scene);
			Camera camera;
			from_json(json.at("camera"), camera);
			return std::make_pair(std::move(scene), std::move(camera));
		}
		catch (nlohmann::json::exception const& e) {
			return TL_ERROR("Failed to deserialize: " + std::string{ e.what() });
		}
	}
}
