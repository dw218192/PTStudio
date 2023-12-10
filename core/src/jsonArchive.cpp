#include "jsonArchive.h"
#include "scene.h"
#include "camera.h"
#include "object.h"
#include "transform.h"
#include "objectID.h"
#include "utils.h"
#include "typeTraitsUtil.h"

#include <unordered_map>
#include <map>
#include <glm/glm.hpp>

namespace {
	// used during deserialization
	// this must only be filled after everything is deserialized and allocated in place
	std::unordered_map<PTS::ObjectID, PTS::Object*> g_id_to_object;
	
	struct {
		auto emplace(PTS::FieldInfo const& info, PTS::ObjectID id) {
			m_data[{ info, m_timestamp++ }] = id;
		}
		auto clear() {
			m_data.clear();
			m_timestamp = 0;
		}
		auto get(PTS::FieldInfo const& info) const -> std::vector<PTS::ObjectID> {
			auto result = std::vector<PTS::ObjectID> {};
			auto it = m_data.lower_bound({ info, 0 });
			while (it != m_data.end() && it->first.first == info) {
				result.emplace_back(it->second);
				++it;
			}
			return result;
		}
		auto count(PTS::FieldInfo const& info) const -> bool {
			auto it = m_data.lower_bound({ info, 0 });
			return it != m_data.end() && it->first.first == info;
		}
	private:
		std::map<std::pair<PTS::FieldInfo, unsigned>, PTS::ObjectID> m_data;
		unsigned m_timestamp = 0;
	} g_pointer_to_id;
	// g_pointer_to_id is filled during deserialization,
	// so we can't map raw pointers to objects because memory might get reallocated
	// g_pointer_to_id is also a multi-map because a field can be a pointer container



	enum { POINTER, POINTER_CONTAINER, NEITHER };
	template<typename T>
	struct is_pointer_or_pointer_container {
		template<typename U>
		static constexpr auto test(int) -> std::enable_if_t<
			PTS::Traits::is_container<U>::value
			&& std::is_pointer_v<typename U::value_type>,
			int
		> { return POINTER_CONTAINER; }
		template<typename U>
		static constexpr auto test(int) -> std::enable_if_t<
			std::is_pointer_v<U>,
			int
		> { return POINTER; }
		template<typename>
		static constexpr auto test(...) -> int { return NEITHER; }
		static constexpr auto value = test<T>(0);
	};

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

	template<typename ObjectOrDerived, int PointerOrPointerContainer, typename TemplatedFieldInfo>
	std::enable_if_t<PTS::Traits::is_reflectable<ObjectOrDerived>::value
		&& std::is_base_of_v<PTS::Object, ObjectOrDerived>>
	from_json_handle_pointers(nlohmann::json const& json, ObjectOrDerived& obj, TemplatedFieldInfo const& info) {
		if constexpr (PointerOrPointerContainer == POINTER) {
			g_pointer_to_id.emplace(PTS::FieldInfo { info, obj }, json.get<PTS::ObjectID>());
		} else if constexpr (PointerOrPointerContainer == POINTER_CONTAINER) {
			for(auto const& id : json) {
				g_pointer_to_id.emplace(PTS::FieldInfo { info, obj }, id.get<PTS::ObjectID>());
			}
		}
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
			if constexpr (field.template has_modifier<MSerialize>()) {
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
			if constexpr (field.template has_modifier<MSerialize>()) {
				if (!json.count(field.var_name)) {
					// use default value if not present in json
					field.get(reflected) = field.get_default();
					return;
				}

				using type = typename decltype(field)::type;
				// handle pointers explicitly, otherwise we can't get the type info
				if constexpr (is_pointer_or_pointer_container<type>::value != NEITHER) {
					from_json_handle_pointers<Reflected, is_pointer_or_pointer_container<type>::value>(
						json.at(field.var_name), reflected, field
					);
				} else {
					from_json(json.at(field.var_name), field.get(reflected));
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
				obj.dyn_for_each_field([&error](FieldInfo const& field) {
					if (g_pointer_to_id.count(field)) {
						auto type = field.type.is_container ? field.type.contained_type() : field.type;

						// ensure that the field is a pointer
						if (!type.is_pointer) {
							error += "Field " + std::string{ field.var_name } + ": is not a pointer, but is in g_pointer_to_id\n";
							return;
						// ensure that the field is a pointer to Object or derived type
						} else if (type != Type::of<Object*>() && !type.pointed_to_type().has_common_base_with(Type::of<Object>())) {
							error += "Field " + std::string{ field.var_name } + ": non-Object pointers are not supported\n";
							return;
						}

						auto field_it = field.begin();
						for (auto id : g_pointer_to_id.get(field)) {
							if (g_id_to_object.count(id)) {
								*static_cast<Object**>(*field_it) = g_id_to_object[id];
							} else if (id == k_invalid_obj_id) {
								*static_cast<Object**>(*field_it) = nullptr;
							} else {
								auto const ptr_name = std::string{ field.type_name } + "::" + std::string{ field.var_name };
								error += "Failed to find object with id " + std::to_string(id) + " for field " + ptr_name + "\n";
							}

							++ field_it;
						}
					}
				});
				if (!error.empty()) {
					return TL_ERROR(error);
				}
			}

			return {};
		} catch (nlohmann::json::exception const& e) {
			return TL_ERROR("Failed to deserialize: " + std::string{ e.what() });
		}
	}
}
