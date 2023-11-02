#pragma once
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include "archive.h"

struct Object;
struct Transform;

struct JsonArchive : Archive {
	auto save(View<Scene> scene_view, View<Camera> camera_view) -> tl::expected<std::string, std::string> override;
	auto load(std::string_view data) -> tl::expected<std::pair<Scene, Camera>, std::string> override;
	auto get_ext() -> std::string_view override { return "json"; }

private:
	nlohmann::json m_json;
};
