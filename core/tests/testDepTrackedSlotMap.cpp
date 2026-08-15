#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/container/depTrackedSlotMap.h>
#include <core/container/slotMap.h>
#include <doctest/doctest.h>

#include <boost/core/span.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using pts::container::DepTrackedSlotMap;
using pts::container::SlotMap;

namespace {

using Span = boost::span<const uint64_t>;

Span make_span(const std::vector<uint64_t>& v) {
    return Span{v.data(), v.size()};
}

}  // namespace

// ===== SlotMap tests =====

TEST_CASE("SlotMap - insert and find") {
    SlotMap<std::string, int> map;
    auto h = map.insert("a", 42);
    CHECK(*h == 42);
    CHECK(h.index() == 0);
    CHECK(map.size() == 1);
    CHECK(map.capacity() == 1);

    auto found = map.find("a");
    CHECK(found);
    CHECK(*found == 42);
    CHECK(found.index() == h.index());

    CHECK(!map.find("b"));
    CHECK(map.contains("a"));
    CHECK(!map.contains("b"));
}

TEST_CASE("SlotMap - upsert bumps version") {
    SlotMap<std::string, int> map;
    auto h = map.insert("a", 10);
    auto v1 = map.version(h);
    map.upsert(h, 20);
    CHECK(*h == 20);
    CHECK(map.version(h) > v1);
}

TEST_CASE("SlotMap - mutate bumps version") {
    SlotMap<std::string, int> map;
    auto h = map.insert("a", 10);
    auto v1 = map.version(h);
    map.mutate(h, [](int& v) { v += 5; });
    CHECK(*h == 15);
    CHECK(map.version(h) > v1);
}

TEST_CASE("SlotMap - erase tombstones and free-list reuses") {
    SlotMap<std::string, int> map;
    auto h1 = map.insert("a", 1);
    auto h2 = map.insert("b", 2);
    auto idx_a = h1.index();

    map.erase("a");
    CHECK(!map.contains("a"));
    CHECK(map.size() == 1);
    CHECK(map.capacity() == 2);

    auto h3 = map.insert("c", 3);
    CHECK(h3.index() == idx_a);
    CHECK(*h3 == 3);
    CHECK(map.size() == 2);
    CHECK(map.capacity() == 2);
}

TEST_CASE("SlotMap - for_each skips inactive") {
    SlotMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);
    map.erase("b");

    std::vector<std::string> keys;
    map.for_each([&](const std::string& k, const int&) { keys.push_back(k); });
    CHECK(keys.size() == 2);
    CHECK(std::find(keys.begin(), keys.end(), "b") == keys.end());
}

TEST_CASE("SlotMap - span_raw includes holes") {
    SlotMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    map.erase("a");

    auto raw = map.span_raw();
    CHECK(raw.size() == 2);
    CHECK(!raw[0].active);
    CHECK(raw[1].active);
    CHECK(raw[1].value == 2);
}

TEST_CASE("SlotMap - at and index-based access") {
    SlotMap<std::string, int> map;
    auto h = map.insert("x", 99);
    CHECK(map.at(h.index()) == 99);
    CHECK(map.active_at(h.index()));
    CHECK(map.version_at(h.index()) > 0);

    map.erase("x");
    CHECK(!map.active_at(h.index()));
}

TEST_CASE("SlotMap - handle stability across reallocation") {
    SlotMap<std::string, int> map;
    auto h0 = map.insert("first", 111);
    CHECK(*h0 == 111);

    for (int i = 0; i < 1024; ++i) {
        map.insert("k" + std::to_string(i), i);
    }

    CHECK(*h0 == 111);
    CHECK(map.at(h0.index()) == 111);
}

TEST_CASE("SlotMap - clear resets") {
    SlotMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    map.clear();
    CHECK(map.size() == 0);
    CHECK(map.capacity() == 0);
    CHECK(!map.contains("a"));
}

TEST_CASE("SlotMap - mutate_at bumps version") {
    SlotMap<std::string, int> map;
    auto h = map.insert("a", 10);
    auto v1 = map.version_at(h.index());
    map.mutate_at(h.index(), [](int& v) { v *= 2; });
    CHECK(map.at(h.index()) == 20);
    CHECK(map.version_at(h.index()) > v1);
}

TEST_CASE("SlotMap - zero heap allocations per entry") {
    struct NoHeap {
        int a = 0;
        float b = 0.0f;
    };
    SlotMap<int, NoHeap> map;
    map.insert(1, NoHeap{42, 3.14f});
    auto raw = map.span_raw();
    CHECK(raw.size() == 1);
    CHECK(raw[0].active);
    CHECK(raw[0].value.a == 42);
    // Entry = {NoHeap, uint64_t, bool} -- all inline, no heap per entry
    static_assert(sizeof(SlotMap<int, NoHeap>::Entry) <= sizeof(NoHeap) + sizeof(uint64_t) + 8,
                  "Entry should have no heap allocation overhead beyond value + version + active");
}

// ===== DepTrackedSlotMap tests =====

TEST_CASE("DepTrackedSlotMap - build on miss") {
    DepTrackedSlotMap<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> deps = {1};
    auto h = cache.get_or_build("a", make_span(deps), [&] {
        ++builds;
        return 42;
    });
    CHECK(*h == 42);
    CHECK(builds == 1);
    CHECK(cache.size() == 1);
    CHECK(cache.version("a") != 0);
}

TEST_CASE("DepTrackedSlotMap - hit with unchanged deps") {
    DepTrackedSlotMap<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> deps = {1, 2, 3};
    cache.get_or_build("a", make_span(deps), [&] {
        ++builds;
        return 100;
    });
    auto v1 = cache.version("a");
    auto h = cache.get_or_build("a", make_span(deps), [&] {
        ++builds;
        return 200;
    });
    CHECK(*h == 100);
    CHECK(builds == 1);
    CHECK(cache.version("a") == v1);
}

TEST_CASE("DepTrackedSlotMap - rebuild on dep change") {
    DepTrackedSlotMap<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> d1 = {1};
    cache.get_or_build("a", make_span(d1), [&] {
        ++builds;
        return 10;
    });
    auto v1 = cache.version("a");

    std::vector<uint64_t> d2 = {2};
    auto h = cache.get_or_build("a", make_span(d2), [&] {
        ++builds;
        return 20;
    });
    CHECK(*h == 20);
    CHECK(builds == 2);
    CHECK(cache.version("a") > v1);
}

TEST_CASE("DepTrackedSlotMap - monotonic versions; independent entries") {
    DepTrackedSlotMap<std::string, int> cache;
    std::vector<uint64_t> d = {};
    cache.get_or_build("a", make_span(d), [] { return 1; });
    cache.get_or_build("b", make_span(d), [] { return 2; });
    cache.get_or_build("c", make_span(d), [] { return 3; });
    CHECK(cache.version("a") < cache.version("b"));
    CHECK(cache.version("b") < cache.version("c"));

    auto vb = cache.version("b");
    auto vc = cache.version("c");
    std::vector<uint64_t> d2 = {99};
    cache.get_or_build("a", make_span(d2), [] { return 10; });
    CHECK(cache.version("b") == vb);
    CHECK(cache.version("c") == vc);
    CHECK(cache.version("a") > vc);
}

TEST_CASE("DepTrackedSlotMap - invalidate forces rebuild") {
    DepTrackedSlotMap<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> d = {1};
    cache.get_or_build("a", make_span(d), [&] {
        ++builds;
        return 1;
    });
    auto v1 = cache.version("a");

    cache.invalidate("a");

    cache.get_or_build("a", make_span(d), [&] {
        ++builds;
        return 2;
    });
    CHECK(builds == 2);
    CHECK(cache.version("a") > v1);

    cache.get_or_build("a", make_span(d), [&] {
        ++builds;
        return 3;
    });
    CHECK(builds == 2);
}

TEST_CASE("DepTrackedSlotMap - erase drops entry; fresh build after") {
    DepTrackedSlotMap<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> d = {1};
    cache.get_or_build("a", make_span(d), [&] {
        ++builds;
        return 1;
    });
    CHECK(cache.contains("a"));
    cache.erase("a");
    CHECK(!cache.contains("a"));
    CHECK(cache.version("a") == 0);

    cache.get_or_build("a", make_span(d), [&] {
        ++builds;
        return 2;
    });
    CHECK(builds == 2);
    CHECK(cache.contains("a"));
}

TEST_CASE("DepTrackedSlotMap - handle stability across unrelated inserts") {
    DepTrackedSlotMap<std::string, int> cache;
    std::vector<uint64_t> d = {};
    auto h_a = cache.get_or_build("a", make_span(d), [] { return 111; });
    CHECK(*h_a == 111);
    for (int i = 0; i < 1024; ++i) {
        cache.get_or_build("k" + std::to_string(i), make_span(d), [i] { return i; });
    }
    CHECK(*h_a == 111);
    auto h_a2 = cache.get_or_build("a", make_span(d), [] { return 999; });
    CHECK(*h_a2 == 111);
}

TEST_CASE("DepTrackedSlotMap - empty deps works") {
    DepTrackedSlotMap<std::string, int> cache;
    int builds = 0;
    Span empty{};
    cache.get_or_build("a", empty, [&] {
        ++builds;
        return 7;
    });
    cache.get_or_build("a", empty, [&] {
        ++builds;
        return 8;
    });
    CHECK(builds == 1);
}

TEST_CASE("DepTrackedSlotMap - cascading invalidation: dep version bumps propagate") {
    DepTrackedSlotMap<std::string, int> bgl;
    DepTrackedSlotMap<std::string, int> pipe;
    DepTrackedSlotMap<std::string, int> desc;

    bgl.get_or_build("layout", Span{}, [] { return 1; });
    auto bgl_v1 = bgl.version("layout");

    uint64_t p_deps1[] = {bgl_v1};
    pipe.get_or_build("pl", Span{p_deps1, 1}, [] { return 10; });
    auto pv1 = pipe.version("pl");

    uint64_t d_deps1[] = {bgl_v1, 42};
    desc.get_or_build("dg", Span{d_deps1, 2}, [] { return 100; });
    auto dv1 = desc.version("dg");

    bgl.invalidate("layout");
    bgl.get_or_build("layout", Span{}, [] { return 2; });
    auto bgl_v2 = bgl.version("layout");
    CHECK(bgl_v2 > bgl_v1);

    uint64_t p_deps2[] = {bgl_v2};
    int rebuilds = 0;
    pipe.get_or_build("pl", Span{p_deps2, 1}, [&] {
        ++rebuilds;
        return 11;
    });
    CHECK(rebuilds == 1);
    CHECK(pipe.version("pl") > pv1);

    uint64_t d_deps2[] = {bgl_v2, 42};
    int desc_rebuilds = 0;
    desc.get_or_build("dg", Span{d_deps2, 2}, [&] {
        ++desc_rebuilds;
        return 101;
    });
    CHECK(desc_rebuilds == 1);
    CHECK(desc.version("dg") > dv1);
}

TEST_CASE("DepTrackedSlotMap - different dep ordering counts as change") {
    DepTrackedSlotMap<std::string, int> cache;
    int builds = 0;
    uint64_t a[] = {1, 2};
    cache.get_or_build("x", Span{a, 2}, [&] {
        ++builds;
        return 1;
    });
    uint64_t b[] = {2, 1};
    cache.get_or_build("x", Span{b, 2}, [&] {
        ++builds;
        return 2;
    });
    CHECK(builds == 2);
}

TEST_CASE("DepTrackedSlotMap - get_or_build_with_replace calls on_replace") {
    DepTrackedSlotMap<std::string, std::unique_ptr<int>> cache;
    int replace_count = 0;
    std::vector<uint64_t> d1 = {1};
    cache.get_or_build_with_replace(
        "a", make_span(d1), [] { return std::make_unique<int>(10); },
        [&](std::unique_ptr<int>& old) {
            ++replace_count;
            old.reset();
        });
    CHECK(replace_count == 0);

    std::vector<uint64_t> d2 = {2};
    auto h = cache.get_or_build_with_replace(
        "a", make_span(d2), [] { return std::make_unique<int>(20); },
        [&](std::unique_ptr<int>& old) {
            ++replace_count;
            old.reset();
        });
    CHECK(replace_count == 1);
    CHECK(**h == 20);
}

TEST_CASE("DepTrackedSlotMap - handle stability across vector reallocation") {
    DepTrackedSlotMap<std::string, int> cache;
    std::vector<uint64_t> d = {};
    auto h0 = cache.get_or_build("first", make_span(d), [] { return 111; });
    CHECK(*h0 == 111);

    for (int i = 0; i < 2048; ++i) {
        cache.get_or_build("k" + std::to_string(i), make_span(d), [i] { return i; });
    }

    CHECK(*h0 == 111);
}

TEST_CASE("DepTrackedSlotMap - transparent lookup with std::less<>") {
    DepTrackedSlotMap<std::string, int, std::less<>> cache;
    std::vector<uint64_t> d = {};
    cache.get_or_build("hello", make_span(d), [] { return 42; });

    std::string_view sv = "hello";
    CHECK(cache.contains(sv));
    CHECK(cache.version(sv) != 0);
    auto h = cache.find(sv);
    CHECK(h);
    CHECK(*h == 42);
}
