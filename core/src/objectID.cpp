#include "objectID.h"

#include <random>
#include <unordered_set>

namespace PTS {
namespace ObjectIDGenerator {
static std::unordered_set<IDType> s_registered_ids;
static std::random_device s_random_device;
static std::mt19937_64 s_random_engine{s_random_device()};
static std::uniform_int_distribution<IDType> s_distribution{1, std::numeric_limits<IDType>::max()};

auto register_id(IDType id) -> void {
    s_registered_ids.insert(id);
}

auto unregister_id(IDType id) -> void {
    if (exists(id)) {
        s_registered_ids.erase(id);
    }
}

auto exists(IDType id) -> bool {
    return s_registered_ids.find(id) != s_registered_ids.end();
}

auto generate_id() -> IDType {
    auto id = IDType{};
    do {
        id = s_distribution(s_random_engine);
    } while (exists(id));
    register_id(id);
    return id;
}
}  // namespace ObjectIDGenerator
}  // namespace PTS