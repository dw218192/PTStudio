#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/container/slotMap.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

using pts::container::SlotMap;

namespace {

/// Test-local field enum + bitmask operators. Lifecycle is intentionally
/// absent -- insert/erase are signaled via SlotArray's any_inserts_for /
/// any_erases_for / drain on_erase callback.
enum class TestField : uint32_t {
    None = 0,
    A = 1u << 0,
    B = 1u << 1,
    C = 1u << 2,
    All = ~0u,
};
constexpr TestField operator|(TestField a, TestField b) noexcept {
    return static_cast<TestField>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr TestField operator&(TestField a, TestField b) noexcept {
    return static_cast<TestField>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr TestField operator~(TestField a) noexcept {
    return static_cast<TestField>(~static_cast<uint32_t>(a));
}

constexpr bool has(TestField mask, TestField bit) {
    return static_cast<uint32_t>(mask & bit) != 0;
}

using TestMap = SlotMap<std::string, int, TestField>;

auto noop_update = [](uint32_t, const int&) {};
auto noop_erase = [](uint32_t) {};

}  // namespace

TEST_CASE("dirty: register_consumer starts fully dirty (forces first-call rebuild)") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B);
    // Even on an empty map, the consumer's first call to any_inserts_for
    // hits true so callers do their initial rebuild path.
    CHECK(map.any_inserts_for(cid));
    CHECK(map.any_dirty_for(cid, TestField::A | TestField::B));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::C));  // outside subscription

    int callbacks = 0;
    map.drain(cid, TestField::All, [&](uint32_t, const int&) { ++callbacks; }, noop_erase);
    CHECK(callbacks == 0);
    CHECK_FALSE(map.any_inserts_for(cid));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));
}

TEST_CASE("dirty: insert primes per-slot dirty for existing consumer") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B);
    map.drain(cid, TestField::All, noop_update, noop_erase);  // clear initial

    map.insert("x", 1);

    CHECK(map.any_inserts_for(cid));
    CHECK(map.any_dirty_for(cid, TestField::A));
    CHECK(map.any_dirty_for(cid, TestField::B));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::C));  // not in subscription
}

TEST_CASE("dirty: register_consumer after inserts sees existing slots dirty") {
    TestMap map;
    map.insert("a", 1);
    map.insert("b", 2);

    auto cid = map.register_consumer(TestField::A);

    CHECK(map.any_inserts_for(cid));
    std::vector<uint32_t> drained;
    map.drain(cid, TestField::A, [&](uint32_t i, const int&) { drained.push_back(i); }, noop_erase);
    CHECK(drained.size() == 2);
}

TEST_CASE("dirty: mutate ORs only the changed bits gated by subscription") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B);

    auto h = map.insert("x", 0);
    map.drain(cid, TestField::All, noop_update, noop_erase);  // clear initial
    REQUIRE_FALSE(map.any_dirty_for(cid, TestField::All));

    // Mutating C alone should NOT mark dirty (C not in subscription).
    map.mutate(h, TestField::C, [](int& v) { v = 42; });
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));

    // Mutating A bit should be visible to consumer.
    map.mutate(h, TestField::A, [](int& v) { v = 7; });
    CHECK(map.any_dirty_for(cid, TestField::A));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::B));
}

TEST_CASE("dirty: any_dirty_for reflects aggregate across slots") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B);

    auto h1 = map.insert("a", 1);
    auto h2 = map.insert("b", 2);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    map.mutate(h1, TestField::A, [](int&) {});
    CHECK(map.any_dirty_for(cid, TestField::A));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::B));

    map.mutate(h2, TestField::B, [](int&) {});
    CHECK(map.any_dirty_for(cid, TestField::A | TestField::B));
}

TEST_CASE("dirty: drain clears only the queried bits") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B | TestField::C);

    auto h = map.insert("x", 0);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    map.mutate(h, TestField::A | TestField::B, [](int&) {});
    REQUIRE(map.any_dirty_for(cid, TestField::A));
    REQUIRE(map.any_dirty_for(cid, TestField::B));

    int callbacks = 0;
    map.drain(cid, TestField::A, [&](uint32_t, const int&) { ++callbacks; }, noop_erase);
    CHECK(callbacks == 1);

    CHECK_FALSE(map.any_dirty_for(cid, TestField::A));
    CHECK(map.any_dirty_for(cid, TestField::B));
}

TEST_CASE("dirty: drain invokes callback with slot index and value") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);

    auto h1 = map.insert("a", 100);
    auto h2 = map.insert("b", 200);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    map.mutate(h1, TestField::A, [](int& v) { v = 111; });
    map.mutate(h2, TestField::A, [](int& v) { v = 222; });

    std::vector<std::pair<uint32_t, int>> seen;
    map.drain(
        cid, TestField::A, [&](uint32_t i, const int& v) { seen.emplace_back(i, v); }, noop_erase);

    CHECK(seen.size() == 2);
    CHECK(seen[0].first == h1.index());
    CHECK(seen[0].second == 111);
    CHECK(seen[1].first == h2.index());
    CHECK(seen[1].second == 222);
}

TEST_CASE("dirty: erase queues an erase event surfaced by drain's on_erase") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);

    auto h = map.insert("x", 1);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    map.erase("x");
    CHECK(map.any_erases_for(cid));

    std::vector<uint32_t> erased;
    map.drain(cid, TestField::All, noop_update, [&](uint32_t i) { erased.push_back(i); });
    CHECK(erased.size() == 1);
    CHECK(erased[0] == h.index());
    CHECK_FALSE(map.any_erases_for(cid));
    CHECK_FALSE(map.active_at(h.index()));
}

TEST_CASE("dirty: drain surfaces erase before update on free-list reuse") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);

    auto h1 = map.insert("a", 1);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    map.erase("a");
    auto h2 = map.insert("b", 2);
    CHECK(h2.index() == h1.index());

    std::vector<std::pair<std::string, uint32_t>> events;
    map.drain(
        cid, TestField::All, [&](uint32_t i, const int&) { events.emplace_back("update", i); },
        [&](uint32_t i) { events.emplace_back("erase", i); });
    REQUIRE(events.size() == 2);
    CHECK(events[0].first == "erase");
    CHECK(events[0].second == h1.index());
    CHECK(events[1].first == "update");
    CHECK(events[1].second == h2.index());
}

TEST_CASE("dirty: two consumers see independent state") {
    TestMap map;
    auto cid_a = map.register_consumer(TestField::A);
    auto cid_b = map.register_consumer(TestField::B);

    auto h = map.insert("x", 0);
    REQUIRE(map.any_dirty_for(cid_a, TestField::A));
    REQUIRE(map.any_dirty_for(cid_b, TestField::B));

    map.drain(cid_a, TestField::A, noop_update, noop_erase);
    CHECK_FALSE(map.any_dirty_for(cid_a, TestField::A));
    CHECK(map.any_dirty_for(cid_b, TestField::B));

    map.mutate(h, TestField::A, [](int&) {});
    CHECK(map.any_dirty_for(cid_a, TestField::A));

    map.drain(cid_b, TestField::B, noop_update, noop_erase);
    CHECK_FALSE(map.any_dirty_for(cid_b, TestField::B));
}

TEST_CASE("dirty: bits outside query survive drain") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B | TestField::C);
    auto h = map.insert("x", 0);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    map.mutate(h, TestField::A | TestField::B | TestField::C, [](int&) {});

    map.drain(cid, TestField::A, noop_update, noop_erase);
    CHECK_FALSE(map.any_dirty_for(cid, TestField::A));
    CHECK(map.any_dirty_for(cid, TestField::B));
    CHECK(map.any_dirty_for(cid, TestField::C));

    map.drain(cid, TestField::B, noop_update, noop_erase);
    CHECK(map.any_dirty_for(cid, TestField::C));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::B));
}

TEST_CASE("dirty: subscription mask filters the propagated bits") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);  // only A

    auto h = map.insert("x", 0);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    map.mutate(h, TestField::A | TestField::B | TestField::C, [](int&) {});

    int callbacks = 0;
    map.drain(cid, TestField::All, [&](uint32_t, const int&) { ++callbacks; }, noop_erase);
    CHECK(callbacks == 1);
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));
}

TEST_CASE("dirty: per-slot bitmap grows as new slots are inserted") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);
    map.insert("a", 1);
    map.drain(cid, TestField::All, noop_update, noop_erase);

    for (int i = 0; i < 10; ++i) {
        map.insert("k" + std::to_string(i), i);
    }
    int n = 0;
    map.drain(cid, TestField::A, [&](uint32_t, const int&) { ++n; }, noop_erase);
    CHECK(n == 10);
}

TEST_CASE("dirty: monostate default disables tracking and remains source-compatible") {
    SlotMap<std::string, int> map;  // DirtyMask = std::monostate (default)
    auto h = map.insert("x", 1);
    map.mutate(h, [](int& v) { v = 42; });
    CHECK(*h == 42);
    CHECK(map.version_at(h.index()) > 0);
}

TEST_CASE("dirty: clear resets per-slot dirty but keeps consumer registration") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);
    map.insert("x", 1);
    map.insert("y", 2);
    REQUIRE(map.any_dirty_for(cid, TestField::A));

    map.clear();
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));
    CHECK_FALSE(map.any_erases_for(cid));
    CHECK_FALSE(map.any_inserts_for(cid));

    map.insert("z", 3);
    CHECK(map.any_inserts_for(cid));
    CHECK(map.any_dirty_for(cid, TestField::A));
}

TEST_CASE("dirty: subscription_for round-trips") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::C);
    auto sub = map.subscription_for(cid);
    CHECK(has(sub, TestField::A));
    CHECK(has(sub, TestField::C));
    CHECK_FALSE(has(sub, TestField::B));
}
