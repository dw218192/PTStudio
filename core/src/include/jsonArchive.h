#pragma once

#include <fstream>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "memory/handle.h"
#include "memory/arena.h"
#include "typeTraitsUtil.h"

namespace PTS {
	struct Object;
	struct Transform;

	struct JsonArchive {
		// id of the arena the object is in
		static constexpr auto k_arena_key = "__arena_id__";
		// id of the arena the object pointed by the handle is in
		static constexpr auto k_handle_arena_key = "__handle_arena_id__";
		// id of the object
		static constexpr auto k_object_key = "__object_id__";
		// id of the object pointed by the handle
		static constexpr auto k_handle_object_key = "__handle_object_id__";

		static constexpr auto k_arena_section = "__arenas__";

		auto open(std::string_view path) -> tl::expected<void, std::string> {
			std::ifstream file {path.data()};
			if (file.is_open()) {
				file >> m_json;
				return {};
			}

			// We deserialize the arena section first,
			// i.e. the actual memory that the objects are stored in,
			// so that it is easier to hook up the references
			
			auto arena_it = m_json.find(k_arena_section);

			return TL_ERROR("Failed to open file {}", path);
		}

		template<typename T>
		struct get_impl {
			static auto do_it(nlohmann::json const& json, std::enable_if_t<Traits::is_reflectable_v<T>, int> _ = 0) -> T {
				T::for_each_field([&json](auto field) {
					using Field = decltype(field);
					using FieldType = typename Field::type;
					if constexpr (std::is_assignable_v<FieldType, nlohmann::json>) {
						field.set(json[field.var_name]);
					} else {
						field.set(get_impl<FieldType>::do_it(json[field.var_name]));
					}
				});
			}
		};
		template<typename T>
		struct set_impl {
			static auto do_it(T const& value, std::enable_if_t<Traits::is_reflectable_v<T>, int> _ = 0) -> nlohmann::json {
				nlohmann::json result;
				T::for_each_field([&result, &value](auto field) {
					using Field = decltype(field);
					using FieldType = typename Field::type;
					if constexpr (std::is_assignable_v<FieldType, nlohmann::json>) {
						result[field.var_name] = field.get(value);
					} else {
						result[field.var_name] = set_impl<FieldType>::do_it(field.get(value));
					}
				});
				return result;
			}
		};
		
		auto get_ext() const -> std::string_view {
			return "json";
		}

		template<typename T>
        auto get(std::string_view key) -> tl::expected<T, std::string> {
			if (auto it = m_json.find(key); it != m_json.end()) {
				if constexpr (std::is_assignable_v<T, nlohmann::json>) {
					return it->get<T>();
				} else {
					return get_impl<T>::do_it(*it);
				}
			}
			return TL_ERROR("Key {} not found in archive", key);
		}

        template<typename T>
        auto set(std::string_view key, T const& value) -> tl::expected<void, std::string> {
			if constexpr (std::is_assignable_v<T, nlohmann::json>) {
				m_json[key] = value;
			} else {
				m_json[key] = set_impl<T>::do_it(value);
			}
		}

		// ----------------- Specializations -----------------
		// glm::vec
		template<glm::length_t L, typename T, glm::qualifier Q>
		struct get_impl<glm::vec<L, T, Q>> {
			static auto do_it(nlohmann::json const& json) -> glm::vec<L, T, Q> {
				glm::vec<L, T, Q> result;
				for (glm::length_t i = 0; i < L; ++i) {
					result[i] = json[i].get<T>();
				}
				return result;
			}
		};
		template<glm::length_t L, typename T, glm::qualifier Q>
		struct set_impl<glm::vec<L, T, Q>> {
			static auto do_it(glm::vec<L, T, Q> const& value) -> nlohmann::json {
				nlohmann::json result;
				for (glm::length_t i = 0; i < L; ++i) {
					result.push_back(value[i]);
				}
				return result;
			}
		};

		// glm::mat
		template<glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
		struct get_impl<glm::mat<C, R, T, Q>> {
			static auto do_it(nlohmann::json const& json) -> glm::mat<C, R, T, Q> {
				glm::mat<C, R, T, Q> result;
				for (glm::length_t i = 0; i < C; ++i) {
					for (glm::length_t j = 0; j < R; ++j) {
						result[i][j] = json[i][j].get<T>();
					}
				}
				return result;
			}
		};

		template<glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
		struct set_impl<glm::mat<C, R, T, Q>> {
			static auto do_it(glm::mat<C, R, T, Q> const& value) -> nlohmann::json {
				nlohmann::json result;
				for (glm::length_t i = 0; i < C; ++i) {
					nlohmann::json row;
					for (glm::length_t j = 0; j < R; ++j) {
						row.push_back(value[i][j]);
					}
					result.push_back(row);
				}
				return result;
			}
		};

		// ObjectHandle
		template<typename T>
		struct get_impl<Handle<T>> {
			static auto do_it(nlohmann::json const& json) -> Handle<T> {
				auto arena_id = json[k_arena_key].get<Arena::IDType>();
				auto id = json["id"].get<ObjectID>();
				auto arena = Arena::get_or_create(arena_id);

				if (arena.is_alive(id)) {
					return Handle<T>{arena.get<T>(id)};
				} else {
					return Handle<T>{};
				}
			}
		};

		template<typename T>
		struct set_impl<Handle<T>> {
			static auto do_it(Handle<T> const& value) -> nlohmann::json {
				nlohmann::json result;
				auto as_base = Handle<Object>{value};

				result[k_arena_key] = as_base->get_arena().get_id();
				result["id"] = as_base->get_id();
			}
		};

	private:
		nlohmann::json m_json;
	};
}
