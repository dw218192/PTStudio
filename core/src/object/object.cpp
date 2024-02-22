#include "object/object.h"
#include "memory/arena.h"

PTS::Object::~Object() noexcept {
}

PTS::Object::Object(std::string_view name) noexcept {
	
}

NODISCARD auto PTS::Object::get_arena() const -> Arena const& {
    return Arena::get_or_create(m_arena_id);
}
