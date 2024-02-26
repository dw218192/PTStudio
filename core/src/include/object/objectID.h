#pragma once

namespace PTS {
    namespace ObjectIDGenerator {
        using IDType = size_t;
        auto register_id(IDType id) -> void;
        auto unregister_id(IDType id) -> void;
        auto exists(IDType id) -> bool;
        auto generate_id() -> IDType;
    } // namespace UUID

    using ObjectID = ObjectIDGenerator::IDType;
    static constexpr auto k_invalid_obj_id = ObjectID { 0 };
} // namespace PTS