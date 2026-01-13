#include <core/legacy/object.h>

PTS::Object::Object(ObjectConstructorUsage usage) noexcept {
    if (usage == ObjectConstructorUsage::DEFAULT) {
        m_id = ObjectIDGenerator::generate_id();
    }  // else m_id will be set by the deserializer

    // usage should ALWAYS be ObjectConstructorUsage::DEFAULT for any other case that is not
    // deserialization
}

PTS::Object::Object(std::string_view name) noexcept : Object{ObjectConstructorUsage::DEFAULT} {
    m_name = name;
}

PTS::Object::~Object() noexcept {
    ObjectIDGenerator::unregister_id(m_id);
}

auto PTS::Object::on_deserialize() noexcept -> void {
    ObjectIDGenerator::register_id(m_id);
}
