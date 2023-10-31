#pragma once
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include "archive.h"

struct Object;
struct Transform;

struct JsonArchive : Archive {
	auto save(View<Scene> scene_view, View<Camera> camera_view) -> tl::expected<std::string, std::string> override;
	auto load() -> tl::expected<std::pair<Scene, Camera>, std::string> override;

private:
	template<glm::length_t L, typename T>
	static auto serialize(glm::vec<L, T> const& vec) -> nlohmann::json {
		nlohmann::json json;
		for (glm::length_t i = 0; i < L; ++i) {
			json.push_back(vec[i]);
		}
		return json;
	}
	template<glm::length_t L, typename T>
	static auto serialize(glm::mat<L, L, T> const& mat) -> nlohmann::json {
		nlohmann::json json;
		for (glm::length_t i = 0; i < L; ++i) {
			json.push_back(serialize(mat[i]));
		}
		return json;
	}

	static auto serialize(Transform const& transform) -> nlohmann::json;
	static auto serialize(Object const& object) -> nlohmann::json;

	// TODO: deserialize
	
	nlohmann::json m_json;
};
