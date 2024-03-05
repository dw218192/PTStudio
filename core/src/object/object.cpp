#include "object/object.h"
#include "memory/arena.h"

PTS::Object::~Object() noexcept = default;

PTS::Object::Object(std::string_view name) noexcept
    : m_name {name} {}

NODISCARD auto PTS::Object::get_arena() const -> Arena const& {
    return Arena::get_or_create_arena(m_arena_id);
}
