#pragma once

namespace PTS {
namespace ObjectIDGenerator {
using IDType = unsigned long long;

auto register_id(IDType id) -> void;
auto unregister_id(IDType id) -> void;
auto exists(IDType id) -> bool;
auto generate_id() -> IDType;
}  // namespace ObjectIDGenerator

using ObjectID = ObjectIDGenerator::IDType;
static constexpr auto k_invalid_obj_id = ObjectID{0};
}  // namespace PTS