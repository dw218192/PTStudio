#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/container/slotMap.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

using pts::container::SlotMap;

namespace {

/// Test-local field enum + bitmask operators.
enum class TestField : uint32_t {
    None = 0,
    Lifecycle = 1u << 0,
    A = 1u << 1,
    B = 1u << 2,
    C = 1u << 3,
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

}  // namespace

TEST_CASE("dirty: register_consumer starts fully dirty (forces first-call rebuild)") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B);
    // Even on an empty map, the consumer's first any_dirty_for hits true so
    // callers always do their initial rebuild path.
    CHECK(map.any_dirty_for(cid, TestField::A | TestField::B));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::C));  // outside subscription

    // Drain clears the synthetic initial dirt; nothing actually iterated.
    int callbacks = 0;
    map.drain_dirty_for(cid, TestField::All, [&](uint32_t, const int&) { ++callbacks; });
    CHECK(callbacks == 0);
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));
}

TEST_CASE("dirty: insert primes per-slot dirty for existing consumer") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::Lifecycle);

    map.insert("x", 1);

    CHECK(map.any_dirty_for(cid, TestField::Lifecycle));
    CHECK(map.any_dirty_for(cid, TestField::A));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::B));  // not in subscription
}

TEST_CASE("dirty: register_consumer after inserts sees existing slots dirty") {
    TestMap map;
    map.insert("a", 1);
    map.insert("b", 2);

    auto cid = map.register_consumer(TestField::A | TestField::Lifecycle);

    CHECK(map.any_dirty_for(cid, TestField::Lifecycle));
    std::vector<uint32_t> drained;
    map.drain_dirty_for(cid, TestField::Lifecycle,
                        [&](uint32_t i, const int&) { drained.push_back(i); });
    CHECK(drained.size() == 2);
}

TEST_CASE("dirty: mutate ORs only the changed bits gated by subscription") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B);

    auto h = map.insert("x", 0);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});  // clear initial
    REQUIRE_FALSE(map.any_dirty_for(cid, TestField::All));

    // Mutating C alone should NOT mark dirty (C not in subscription)
    map.mutate(h, TestField::C, [](int& v) { v = 42; });
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));

    // Mutating A bit should be visible to consumer
    map.mutate(h, TestField::A, [](int& v) { v = 7; });
    CHECK(map.any_dirty_for(cid, TestField::A));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::B));
}

TEST_CASE("dirty: any_dirty_for reflects aggregate across slots") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B);

    auto h1 = map.insert("a", 1);
    auto h2 = map.insert("b", 2);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    map.mutate(h1, TestField::A, [](int&) {});
    CHECK(map.any_dirty_for(cid, TestField::A));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::B));

    map.mutate(h2, TestField::B, [](int&) {});
    CHECK(map.any_dirty_for(cid, TestField::A | TestField::B));
}

TEST_CASE("dirty: drain_dirty_for clears only the queried bits") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B | TestField::C);

    auto h = map.insert("x", 0);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    map.mutate(h, TestField::A | TestField::B, [](int&) {});
    REQUIRE(map.any_dirty_for(cid, TestField::A));
    REQUIRE(map.any_dirty_for(cid, TestField::B));

    int callbacks = 0;
    map.drain_dirty_for(cid, TestField::A, [&](uint32_t, const int&) { ++callbacks; });
    CHECK(callbacks == 1);

    // A is cleared, B remains
    CHECK_FALSE(map.any_dirty_for(cid, TestField::A));
    CHECK(map.any_dirty_for(cid, TestField::B));
}

TEST_CASE("dirty: drain_dirty_for invokes callback with slot index and value") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);

    auto h1 = map.insert("a", 100);
    auto h2 = map.insert("b", 200);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    map.mutate(h1, TestField::A, [](int& v) { v = 111; });
    map.mutate(h2, TestField::A, [](int& v) { v = 222; });

    std::vector<std::pair<uint32_t, int>> seen;
    map.drain_dirty_for(cid, TestField::A,
                        [&](uint32_t i, const int& v) { seen.emplace_back(i, v); });

    CHECK(seen.size() == 2);
    CHECK(seen[0].first == h1.index());
    CHECK(seen[0].second == 111);
    CHECK(seen[1].first == h2.index());
    CHECK(seen[1].second == 222);
}

TEST_CASE("dirty: erase marks slot dirty for consumers (lifecycle observation)") {
    TestMap map;
    auto cid = map.register_consumer(TestField::Lifecycle);

    auto h = map.insert("x", 1);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    map.erase("x");
    CHECK(map.any_dirty_for(cid, TestField::Lifecycle));

    // Drained slot is now inactive
    bool saw_inactive = false;
    map.drain_dirty_for(cid, TestField::Lifecycle, [&](uint32_t i, const int&) {
        if (!map.active_at(i)) saw_inactive = true;
    });
    CHECK(saw_inactive);

    UNUSED(h);
}

TEST_CASE("dirty: two consumers see independent state") {
    TestMap map;
    auto cid_a = map.register_consumer(TestField::A);
    auto cid_b = map.register_consumer(TestField::B);

    auto h = map.insert("x", 0);
    // Both consumers see the lifecycle change as their respective subscription bit.
    REQUIRE(map.any_dirty_for(cid_a, TestField::A));
    REQUIRE(map.any_dirty_for(cid_b, TestField::B));

    // Drain consumer A only.
    map.drain_dirty_for(cid_a, TestField::A, [](uint32_t, const int&) {});
    CHECK_FALSE(map.any_dirty_for(cid_a, TestField::A));
    // Consumer B still has its bit.
    CHECK(map.any_dirty_for(cid_b, TestField::B));

    // Mutate A bit -- only consumer A subscribed.
    map.mutate(h, TestField::A, [](int&) {});
    CHECK(map.any_dirty_for(cid_a, TestField::A));

    // Consumer B unaffected by A-only mutation.
    map.drain_dirty_for(cid_b, TestField::B, [](uint32_t, const int&) {});
    CHECK_FALSE(map.any_dirty_for(cid_b, TestField::B));
}

TEST_CASE("dirty: bits outside query survive drain") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::B | TestField::C);
    auto h = map.insert("x", 0);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    map.mutate(h, TestField::A | TestField::B | TestField::C, [](int&) {});

    // Drain only A; B and C should remain.
    map.drain_dirty_for(cid, TestField::A, [](uint32_t, const int&) {});
    CHECK_FALSE(map.any_dirty_for(cid, TestField::A));
    CHECK(map.any_dirty_for(cid, TestField::B));
    CHECK(map.any_dirty_for(cid, TestField::C));

    // Drain B -- only C remains.
    map.drain_dirty_for(cid, TestField::B, [](uint32_t, const int&) {});
    CHECK(map.any_dirty_for(cid, TestField::C));
    CHECK_FALSE(map.any_dirty_for(cid, TestField::B));
}

TEST_CASE("dirty: subscription mask filters the propagated bits") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A);  // only A

    auto h = map.insert("x", 0);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    // Mutate sets A | B | C -- consumer should only see A.
    map.mutate(h, TestField::A | TestField::B | TestField::C, [](int&) {});

    int callbacks = 0;
    map.drain_dirty_for(cid, TestField::All, [&](uint32_t, const int&) { ++callbacks; });
    CHECK(callbacks == 1);  // one slot, drained once

    // No bits should remain after draining All
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));
}

TEST_CASE("dirty: per-slot bitmap grows as new slots are inserted") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::Lifecycle);
    map.insert("a", 1);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    // Insert more slots; growth should not crash and new slots should be dirty.
    for (int i = 0; i < 10; ++i) {
        map.insert("k" + std::to_string(i), i);
    }
    int n = 0;
    map.drain_dirty_for(cid, TestField::Lifecycle, [&](uint32_t, const int&) { ++n; });
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
    auto cid = map.register_consumer(TestField::A | TestField::Lifecycle);
    map.insert("x", 1);
    map.insert("y", 2);
    REQUIRE(map.any_dirty_for(cid, TestField::Lifecycle));

    map.clear();
    // After clear, per-slot dirty is wiped and aggregate is zero -- the
    // consumer is still registered, but sees a fresh (clean) map.
    CHECK_FALSE(map.any_dirty_for(cid, TestField::All));

    // Subsequent insert should still propagate to the consumer.
    map.insert("z", 3);
    CHECK(map.any_dirty_for(cid, TestField::Lifecycle));
}

TEST_CASE("dirty: free-list slot reuse re-marks lifecycle dirty") {
    TestMap map;
    auto cid = map.register_consumer(TestField::Lifecycle);

    auto h = map.insert("a", 1);
    map.drain_dirty_for(cid, TestField::All, [](uint32_t, const int&) {});

    map.erase("a");
    map.drain_dirty_for(cid, TestField::Lifecycle, [](uint32_t, const int&) {});

    auto h2 = map.insert("b", 2);
    CHECK(h2.index() == h.index());  // free-list reuse
    CHECK(map.any_dirty_for(cid, TestField::Lifecycle));
}

TEST_CASE("dirty: subscription_for round-trips") {
    TestMap map;
    auto cid = map.register_consumer(TestField::A | TestField::C);
    auto sub = map.subscription_for(cid);
    CHECK(has(sub, TestField::A));
    CHECK(has(sub, TestField::C));
    CHECK_FALSE(has(sub, TestField::B));
}
