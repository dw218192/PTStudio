#pragma once
#include <unordered_map>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include "archive.h"

namespace PTS {
	struct Object;
	struct Transform;

	struct JsonArchive : Archive {
		template <typename Reflected, typename>
		friend auto from_json(nlohmann::json const& json, Reflected& reflected) -> void;
		template <typename Managed, typename>
		friend auto from_json(nlohmann::json const& json, ViewPtr<Managed>& ptr) -> void;
		auto save(View<Scene> scene_view, View<Camera> camera_view) -> tl::expected<std::string, std::string> override;
		auto load(std::string_view data, Ref<Scene> scene, Ref<Camera> cam) -> tl::expected<void, std::string> override;
		auto get_ext() -> std::string_view override { return "json"; }

	private:
		nlohmann::json m_json;
	};
}
