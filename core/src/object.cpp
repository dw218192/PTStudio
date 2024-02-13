#include "object.h"
#include "archive.h"

PTS::Object::~Object() noexcept {
}

auto PTS::Object::serialize(Archive& archive) const -> void {
	archive.set("name", m_name);
	archive.set("id", m_id);
}

auto PTS::Object::on_deserialize() noexcept -> void {
}
