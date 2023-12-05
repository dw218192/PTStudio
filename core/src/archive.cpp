#include "archive.h"
#include <fstream>

auto PTS::Archive::load_file(std::string_view file, Ref<Scene> scene, Ref<Camera> cam) noexcept -> tl::expected<void, std::string> {
	std::ifstream stream { file.data() };
	if (!stream.is_open()) {
		return TL_ERROR("Failed to open archive file " + std::string{ file });
	}
	std::string const data{ std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
	return load(data, scene, cam);
}

auto PTS::Archive::save_file(View<Scene> scene_view, View<Camera> camera_view, std::string_view file) noexcept -> tl::expected<void, std::string> {
	std::ofstream stream{ file.data() };
	if (!stream.is_open()) {
		return TL_ERROR("Failed to open archive file " + std::string{ file });
	}
	std::string data;
	TL_TRY_ASSIGN(data, save(scene_view, camera_view));
	if (!(stream << data)) {
		return TL_ERROR("Failed to write to file " + std::string{ file });
	}
	return {};
}
