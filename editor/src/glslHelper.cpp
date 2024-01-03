#include "include/glslHelper.h"
#include "include/editorResources.h"
#include "embeddedRes.h"

#include <regex>
#include <sstream>
#include <optional>

using namespace PTS;

static constexpr auto k_inc_search_paths = std::array{
	"",
	"shaders/",
	"shaders/include/",
	"shaders/user/",
};

auto GLSLHelper::preprocess(std::string_view src) -> std::string {
	auto const reg = std::regex(R"(\s*\#\s*include\s*<([^<>]+)>|\s*\#\s*include\s*\"([^"]+)\")");
	auto matches = std::match_results<std::string_view::const_iterator>{};

	struct Snippet {
		enum class Type { Text, RelInc, AbsInc } type;

		std::string text;
		Snippet(Snippet::Type type, std::string text) : type{type}, text{std::move(text)} {}
	};
	auto snippets = std::vector<Snippet>{};
	auto start = src.cbegin();
	for (; std::regex_search(start, src.cend(), matches, reg);
	       start = matches.suffix().first) {
		snippets.emplace_back(Snippet::Type::Text, std::string{start, matches[0].first});

		if (matches[1].matched) {
			snippets.emplace_back(Snippet::Type::AbsInc, matches[1].str());
		} else if (matches[2].matched) {
			snippets.emplace_back(Snippet::Type::RelInc, matches[2].str());
		}
	}
	snippets.emplace_back(Snippet::Type::Text, std::string{start, src.cend()});

	auto ret = std::string{};
	for (auto const& snippet : snippets) {
		if (snippet.text.empty()) {
			continue;
		}

		auto inc_src = std::string{};
		auto fs = std::optional<cmrc::embedded_filesystem>{};
		if (snippet.type == Snippet::Type::AbsInc) {
			fs = cmrc::core_resources::get_filesystem();
		} else if (snippet.type == Snippet::Type::RelInc) {
			fs = cmrc::editor_resources::get_filesystem();
		} else {
			ret.append(snippet.text);
			continue;
		}

		// try all search paths, stop at the first match
		for (auto const& path : k_inc_search_paths) {
			if (auto res = try_get_embedded_res(fs.value(), path + snippet.text)) {
				inc_src = std::move(res.value());
				break;
			}
		}

		if (inc_src.empty()) {
			return fmt::format("include file {} not found", snippet.text);
		} else {
			ret.push_back('\n');
			ret.append(preprocess(inc_src));
		}
	}
	return ret;
}
