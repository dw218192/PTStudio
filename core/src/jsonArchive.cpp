#include "jsonArchive.h"
#include "scene.h"
#include "camera.h"
#include "object.h"
#include "transform.h"
#include "objectID.h"
#include "utils.h"
#include "typeTraitsUtil.h"

#include <iostream>
#include <glm/glm.hpp>

namespace {
	// used during deserialization
	// this must only be filled after everything is deserialized and allocated in place
	std::unordered_map<PTS::ObjectID, PTS::Object*> g_id_to_object;
	// this is filled during deserialization,
	// so we can't map raw pointers to objects because memory might get reallocated
	std::unordered_map<PTS::FieldInfo, PTS::ObjectID> g_pointer_to_id;

	template<typename ObjectOrDerived>
	std::enable_if_t<PTS::Traits::is_reflectable<ObjectOrDerived>::value
		&& std::is_base_of_v<PTS::Object, ObjectOrDerived>>
	gather_objects(ObjectOrDerived& obj) {
		g_id_to_object[obj.get_id()] = &obj;

		ObjectOrDerived::for_each_field([&obj](auto field) {
			using type = typename decltype(field)::type;
			if constexpr (PTS::Traits::is_container<type>::value) {
				using value_type = typename std::iterator_traits<typename type::iterator>::value_type;
				if constexpr (PTS::Traits::is_reflectable<value_type>::value && std::is_base_of_v<PTS::Object, value_type>) {
					for (auto& elem : field.get(obj)) {
						gather_objects(elem);
					}
				}
			} else if constexpr (PTS::Traits::is_reflectable<type>::value && std::is_base_of_v<PTS::Object, type>) {
				gather_objects(field.get(obj));
			}
		});
	}
}

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
	// serialization and deserialization for pointers and references
	// Note: only pointers to objects or derived types are supported
	// because otherwise we can't do dynamic reflection & uuid
	template<typename ObjectOrDerived>
	std::enable_if_t<std::is_base_of_v<Object, ObjectOrDerived>>
	to_json(nlohmann::json& json, ViewPtr<ObjectOrDerived> ptr) {
		if (!ptr) {
			json = k_invalid_obj_id;
		} else {
			json = ptr->get_id();
		}
	}
	template<typename ObjectOrDerived>
	std::enable_if_t<std::is_base_of_v<Object, ObjectOrDerived>>
	to_json(nlohmann::json& json, ObserverPtr<ObjectOrDerived> ptr) {
		to_json(json, ViewPtr<Object> { ptr });
	}

	// from_json for pointers are handled in from_json(nlohmann::json const& json, Reflected& reflected)

	template<typename Reflected>
	std::enable_if_t<Traits::is_reflectable<Reflected>::value>
	to_json(nlohmann::json& json, Reflected const& reflected) {
		if constexpr (Traits::has_serialization_callback<Reflected>::value) {
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
	template<typename Reflected>
	std::enable_if_t<Traits::is_reflectable<Reflected>::value>
	from_json(nlohmann::json const& json, Reflected& reflected) {
		Reflected::for_each_field([&reflected, &json](auto field) {
			if (field.template get_modifier<MSerialize>()) {

				// handle pointers explicitly, otherwise we can't get the type info
				if constexpr (std::is_pointer_v<typename decltype(field)::type>) {
					if (json.count(field.var_name)) {
						auto const id = json.at(field.var_name).get<ObjectID>();
						if (id == k_invalid_obj_id) {
							field.get(reflected) = nullptr;
						} else {
							// register the pointer to id mapping
							g_pointer_to_id[FieldInfo{field}] = id;
						}
					} else {
						field.get(reflected) = nullptr;
					}
				} else {
					if (json.count(field.var_name)) {
						from_json(json.at(field.var_name), field.get(reflected));
					} else {
						// use default value if not present in json
						field.get(reflected) = field.get_default();
					}
				}
			}
		});
		if constexpr (Traits::has_deserialization_callback<Reflected>::value) {
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
		} catch (nlohmann::json::exception const& e) {
			return TL_ERROR("Failed to serialize: " + std::string{ e.what() });
		}

		return json.dump();
	}

	auto JsonArchive::load(std::string_view data, Ref<Scene> scene, Ref<Camera> cam) -> tl::expected<void, std::string> {
		try {
			auto json = nlohmann::json::parse(data);
	
			g_id_to_object.clear();
			g_pointer_to_id.clear();

			from_json(json.at("scene"), scene.get());
			from_json(json.at("camera"), cam.get());
			// gather all objects
			gather_objects(scene.get());

			// handle pointers and references
			for (auto kvp : g_id_to_object) {
				auto error = std::string {};
				auto& obj = *kvp.second;
				dynamic_for_each_field(obj, [this, &obj, &error](FieldInfo const& field) {
					if (field.is_pointer) {
						// should be ok to treat Object* T::* as U* T::* where U is a derived type of Object
						// TODO: check the standard
						auto&& ptr = field.get<Object*>(&obj);
						auto const ptr_name = std::string{ field.type_name } + "::" + std::string{ field.var_name };
						if (g_pointer_to_id.count(field)) {
							if (g_id_to_object.count(g_pointer_to_id.at(field))) {
								ptr = g_id_to_object.at(g_pointer_to_id.at(field));
							} else {
								error += "Failed to deserialize: destination object for " + ptr_name + " not found";
								error.push_back('\n');
							}
						} else {
							error += "Failed to deserialize: " + ptr_name + " not found";
							error.push_back('\n');
						}
					}
				});
				if (!error.empty()) {
					return TL_ERROR(error);
				}
			}

			return {};
		}
		catch (nlohmann::json::exception const& e) {
			return TL_ERROR("Failed to deserialize: " + std::string{ e.what() });
		}
	}
}
