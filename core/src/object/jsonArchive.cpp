#include "object/jsonArchive.h"

namespace {
	auto try_get(nlohmann::json entry, std::string_view key) -> nlohmann::json::iterator {
		auto const it = entry.find(key);
		if (it == entry.end()) {
			
		}
	}
}


auto PTS::JsonArchive::open(std::string_view path) -> tl::expected<void, std::string> {
	std::ifstream file{ path.data() };
	if (file.is_open()) {
		file >> m_json;
	}
	else {
		return TL_ERROR("Failed to open file {}", path);
	}

	// We deserialize the arenas first,
	// i.e. the actual memory that the objects are stored in,
	// so that it is easier to hook up the references
	auto const it = m_json.find(k_arena_section_key);
	if (it != m_json.end()) {
		// for each arena
		for (auto arena_it = it->begin(); arena_it != it->end(); ++arena_it) {
			auto const arena_id_it = arena_it->find(k_arena_id_key);
			if (arena_id_it == arena_it->end()) {
				return TL_ERROR("Syntax error in file {}, expected {}", path, k_arena_objects_key);
			}

			auto const arena_id = arena_id_it->get<ArenaID>();
			auto& arena = Arena::get_or_create_arena(arena_id);

			auto const objects_it = arena_it->find(k_arena_objects_key);
			if (objects_it == arena_it->end()) {
				return TL_ERROR("Syntax error in file {}, expected {}", path, k_arena_objects_key);
			}

			for (auto obj_it = objects_it->begin(); obj_it != objects_it->end(); ++obj_it) {
				auto creator_fn = PTS::ObjectRegistry::CreatorFn{};

			}
		}
	}
}