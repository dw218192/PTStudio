#pragma once

#include <type_traits>
#include <unordered_map>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

namespace PTS {
	struct Object;
	struct Transform;

	struct JsonArchive {
		template<typename T>
		struct get_impl {
			static auto do_it(nlohmann::json const& json) -> T;
		};
		template<typename T>
		struct set_impl {
			static auto do_it(T const& value) -> nlohmann::json;
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
		

	private:
		nlohmann::json m_json;
	};
}
