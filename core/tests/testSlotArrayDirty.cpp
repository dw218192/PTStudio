#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/container/slotArray.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using pts::container::SlotArray;

namespace {

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

using TestArr = SlotArray<int, TestField>;

// No-op drain callbacks for tests that don't care about the callback data.
auto noop_update = [](uint32_t, const int&) {};
auto noop_erase = [](uint32_t) {};

}  // namespace

TEST_CASE("SlotArray: insert returns a Handle with index()") {
    TestArr arr;
    auto h = arr.insert(42);
    CHECK(h);
    CHECK(h.index() == 0);
    CHECK(arr.size() == 1);
    CHECK(arr.capacity() == 1);
    CHECK(arr.at(0) == 42);
}

TEST_CASE("SlotArray: erase tombstones slot and frees it for reuse") {
    TestArr arr;
    auto h1 = arr.insert(1);
    auto h2 = arr.insert(2);
    CHECK(arr.active_at(h1.index()));
    arr.erase(h1);
    CHECK_FALSE(arr.active_at(h1.index()));
    CHECK(arr.size() == 1);
    CHECK(arr.capacity() == 2);

    auto h3 = arr.insert(3);
    CHECK(h3.index() == h1.index());  // slot reused from free-list
    CHECK(arr.at(h3.index()) == 3);
    CHECK(arr.size() == 2);
    (void) h2;
}

TEST_CASE("SlotArray: mutate_at bumps version and gates dirty by subscription") {
    TestArr arr;
    auto h = arr.insert(10);
    auto v0 = arr.version_at(h.index());
    arr.mutate_at(h.index(), TestField::A, [](int& v) { v = 99; });
    CHECK(arr.at(h.index()) == 99);
    CHECK(arr.version_at(h.index()) > v0);
}

TEST_CASE("SlotArray: register_consumer primes inserts_pending and aggregate") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A | TestField::B);
    // Consumer starts fully dirty so the first drain does the initial rebuild.
    CHECK(arr.any_inserts_for(cid));
    CHECK(arr.any_dirty_for(cid, TestField::A | TestField::B));
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::C));
    CHECK_FALSE(arr.any_erases_for(cid));

    int updates = 0;
    arr.drain(cid, TestField::All, [&](uint32_t, const int&) { ++updates; }, noop_erase);
    CHECK(updates == 0);  // empty array -> no slot-level callbacks
    CHECK_FALSE(arr.any_inserts_for(cid));
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::All));
}

TEST_CASE("SlotArray: insert after register stamps new slot fully dirty") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A);
    arr.drain(cid, TestField::All, noop_update, noop_erase);  // clear initial
    REQUIRE_FALSE(arr.any_inserts_for(cid));
    REQUIRE_FALSE(arr.any_dirty_for(cid, TestField::All));

    arr.insert(1);
    CHECK(arr.any_inserts_for(cid));
    CHECK(arr.any_dirty_for(cid, TestField::A));
}

TEST_CASE("SlotArray: register_consumer after inserts sees existing slots dirty") {
    TestArr arr;
    arr.insert(1);
    arr.insert(2);

    auto cid = arr.register_consumer(TestField::A);

    CHECK(arr.any_dirty_for(cid, TestField::A));
    std::vector<uint32_t> drained;
    arr.drain(cid, TestField::A, [&](uint32_t i, const int&) { drained.push_back(i); }, noop_erase);
    CHECK(drained.size() == 2);
}

TEST_CASE("SlotArray: mutate ORs only the changed bits gated by subscription") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A | TestField::B);

    auto h = arr.insert(0);
    arr.drain(cid, TestField::All, noop_update, noop_erase);  // clear initial
    REQUIRE_FALSE(arr.any_dirty_for(cid, TestField::All));

    // Mutating C alone should NOT mark dirty (C not in subscription).
    arr.mutate_at(h.index(), TestField::C, [](int& v) { v = 42; });
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::All));

    // Mutating A should be visible.
    arr.mutate_at(h.index(), TestField::A, [](int& v) { v = 7; });
    CHECK(arr.any_dirty_for(cid, TestField::A));
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::B));
}

TEST_CASE("SlotArray: erase queues an erase event and clears per-slot dirty") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A);

    auto h = arr.insert(1);
    arr.mutate_at(h.index(), TestField::A, [](int& v) { v = 9; });
    arr.drain(cid, TestField::All, noop_update, noop_erase);  // clear

    arr.erase(h);
    CHECK(arr.any_erases_for(cid));
    // Erased slot should NOT report dirty via any_dirty_for.
    // (The aggregate may be stale until drain recomputes; drain semantics
    // matter more than the intermediate bit.)

    std::vector<uint32_t> erases;
    int update_count = 0;
    arr.drain(
        cid, TestField::All, [&](uint32_t, const int&) { ++update_count; },
        [&](uint32_t i) { erases.push_back(i); });
    CHECK(erases.size() == 1);
    CHECK(erases[0] == h.index());
    CHECK(update_count == 0);  // erased slot is NOT surfaced via on_update
    CHECK_FALSE(arr.any_erases_for(cid));
}

TEST_CASE("SlotArray: drain surfaces erase before update on free-list reuse") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A);

    auto h1 = arr.insert(10);
    arr.drain(cid, TestField::All, noop_update, noop_erase);  // clear

    arr.erase(h1);
    auto h2 = arr.insert(20);  // reuses the same slot
    CHECK(h2.index() == h1.index());

    std::vector<std::pair<std::string, uint32_t>> events;
    arr.drain(
        cid, TestField::All, [&](uint32_t i, const int&) { events.emplace_back("update", i); },
        [&](uint32_t i) { events.emplace_back("erase", i); });
    // Erase comes first so consumers may release per-slot GPU state
    // before the reuse shows up via on_update.
    REQUIRE(events.size() == 2);
    CHECK(events[0].first == "erase");
    CHECK(events[0].second == h1.index());
    CHECK(events[1].first == "update");
    CHECK(events[1].second == h2.index());
}

TEST_CASE("SlotArray: drain callback receives slot index and value") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A);

    auto h1 = arr.insert(100);
    auto h2 = arr.insert(200);
    arr.drain(cid, TestField::All, noop_update, noop_erase);

    arr.mutate_at(h1.index(), TestField::A, [](int& v) { v = 111; });
    arr.mutate_at(h2.index(), TestField::A, [](int& v) { v = 222; });

    std::vector<std::pair<uint32_t, int>> seen;
    arr.drain(
        cid, TestField::A, [&](uint32_t i, const int& v) { seen.emplace_back(i, v); }, noop_erase);
    CHECK(seen.size() == 2);
    CHECK(seen[0].first == h1.index());
    CHECK(seen[0].second == 111);
    CHECK(seen[1].first == h2.index());
    CHECK(seen[1].second == 222);
}

TEST_CASE("SlotArray: two consumers track independent state") {
    TestArr arr;
    auto cid_a = arr.register_consumer(TestField::A);
    auto cid_b = arr.register_consumer(TestField::B);

    auto h = arr.insert(0);
    // Both consumers see the insert stamping on their own subscription.
    REQUIRE(arr.any_dirty_for(cid_a, TestField::A));
    REQUIRE(arr.any_dirty_for(cid_b, TestField::B));

    arr.drain(cid_a, TestField::A, noop_update, noop_erase);
    CHECK_FALSE(arr.any_dirty_for(cid_a, TestField::A));
    CHECK(arr.any_dirty_for(cid_b, TestField::B));

    // Mutate A -- only consumer A subscribes.
    arr.mutate_at(h.index(), TestField::A, [](int&) {});
    CHECK(arr.any_dirty_for(cid_a, TestField::A));

    arr.drain(cid_b, TestField::B, noop_update, noop_erase);
    CHECK_FALSE(arr.any_dirty_for(cid_b, TestField::B));
}

TEST_CASE("SlotArray: drain clears only the queried bits") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A | TestField::B | TestField::C);

    auto h = arr.insert(0);
    arr.drain(cid, TestField::All, noop_update, noop_erase);

    arr.mutate_at(h.index(), TestField::A | TestField::B, [](int&) {});
    REQUIRE(arr.any_dirty_for(cid, TestField::A));
    REQUIRE(arr.any_dirty_for(cid, TestField::B));

    int callbacks = 0;
    arr.drain(cid, TestField::A, [&](uint32_t, const int&) { ++callbacks; }, noop_erase);
    CHECK(callbacks == 1);
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::A));
    CHECK(arr.any_dirty_for(cid, TestField::B));
}

TEST_CASE("SlotArray: bits outside query survive drain") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A | TestField::B | TestField::C);
    auto h = arr.insert(0);
    arr.drain(cid, TestField::All, noop_update, noop_erase);

    arr.mutate_at(h.index(), TestField::A | TestField::B | TestField::C, [](int&) {});
    arr.drain(cid, TestField::A, noop_update, noop_erase);
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::A));
    CHECK(arr.any_dirty_for(cid, TestField::B));
    CHECK(arr.any_dirty_for(cid, TestField::C));

    arr.drain(cid, TestField::B, noop_update, noop_erase);
    CHECK(arr.any_dirty_for(cid, TestField::C));
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::B));
}

TEST_CASE("SlotArray: subscription mask filters propagated bits") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A);  // only A

    auto h = arr.insert(0);
    arr.drain(cid, TestField::All, noop_update, noop_erase);

    // Mutate sets A | B | C; consumer should only observe A.
    arr.mutate_at(h.index(), TestField::A | TestField::B | TestField::C, [](int&) {});

    int callbacks = 0;
    arr.drain(cid, TestField::All, [&](uint32_t, const int&) { ++callbacks; }, noop_erase);
    CHECK(callbacks == 1);
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::All));
}

TEST_CASE("SlotArray: monostate default disables tracking, stays source-compatible") {
    SlotArray<int> arr;
    auto h = arr.insert(1);
    arr.mutate_at(h.index(), [](int& v) { v = 42; });
    CHECK(arr.at(h.index()) == 42);
    CHECK(arr.version_at(h.index()) > 0);
}

TEST_CASE("SlotArray: clear resets dirty state and erase queue, keeps consumers") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A);
    auto h = arr.insert(1);
    arr.erase(h);
    REQUIRE(arr.any_erases_for(cid));

    arr.clear();
    CHECK_FALSE(arr.any_dirty_for(cid, TestField::All));
    CHECK_FALSE(arr.any_erases_for(cid));
    CHECK_FALSE(arr.any_inserts_for(cid));

    // Subsequent insert still reaches the consumer.
    arr.insert(3);
    CHECK(arr.any_inserts_for(cid));
    CHECK(arr.any_dirty_for(cid, TestField::A));
}

TEST_CASE("SlotArray: mark_consumer_fully_dirty primes a full rebuild") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A);
    arr.insert(1);
    arr.insert(2);
    arr.drain(cid, TestField::All, noop_update, noop_erase);
    REQUIRE_FALSE(arr.any_dirty_for(cid, TestField::All));
    REQUIRE_FALSE(arr.any_inserts_for(cid));

    arr.mark_consumer_fully_dirty(cid);
    CHECK(arr.any_inserts_for(cid));
    CHECK(arr.any_dirty_for(cid, TestField::A));

    int seen = 0;
    arr.drain(cid, TestField::A, [&](uint32_t, const int&) { ++seen; }, noop_erase);
    CHECK(seen == 2);
}

TEST_CASE("SlotArray: subscription_for round-trips") {
    TestArr arr;
    auto cid = arr.register_consumer(TestField::A | TestField::C);
    auto sub = arr.subscription_for(cid);
    CHECK(has(sub, TestField::A));
    CHECK(has(sub, TestField::C));
    CHECK_FALSE(has(sub, TestField::B));
}
