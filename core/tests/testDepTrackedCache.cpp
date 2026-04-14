#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/cache/depTrackedCache.h>
#include <doctest/doctest.h>

#include <boost/core/span.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using pts::cache::DepTrackedCache;

namespace {

using Span = boost::span<const uint64_t>;

Span make_span(const std::vector<uint64_t>& v) {
    return Span{v.data(), v.size()};
}

}  // namespace

TEST_CASE("DepTrackedCache - build on miss") {
    DepTrackedCache<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> deps = {1};
    const int& v = cache.get_or_build("a", make_span(deps), [&] {
        ++builds;
        return 42;
    });
    CHECK(v == 42);
    CHECK(builds == 1);
    CHECK(cache.size() == 1);
    CHECK(cache.version("a") != 0);
}

TEST_CASE("DepTrackedCache - hit with unchanged deps") {
    DepTrackedCache<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> deps = {1, 2, 3};
    cache.get_or_build("a", make_span(deps), [&] {
        ++builds;
        return 100;
    });
    auto v1 = cache.version("a");
    const int& v = cache.get_or_build("a", make_span(deps), [&] {
        ++builds;
        return 200;
    });
    CHECK(v == 100);
    CHECK(builds == 1);
    CHECK(cache.version("a") == v1);
}

TEST_CASE("DepTrackedCache - rebuild on dep change") {
    DepTrackedCache<std::string, int> cache;
    int builds = 0;
    std::vector<uint64_t> d1 = {1};
    cache.get_or_build("a", make_span(d1), [&] {
        ++builds;
        return 10;
    });
    auto v1 = cache.version("a");

    std::vector<uint64_t> d2 = {2};
    const int& v = cache.get_or_build("a", make_span(d2), [&] {
        ++builds;
        return 20;
    });
    CHECK(v == 20);
    CHECK(builds == 2);
    CHECK(cache.version("a") > v1);
}

TEST_CASE("DepTrackedCache - monotonic versions; independent entries") {
    DepTrackedCache<std::string, int> cache;
    std::vector<uint64_t> d = {};
    cache.get_or_build("a", make_span(d), [] { return 1; });
    cache.get_or_build("b", make_span(d), [] { return 2; });
    cache.get_or_build("c", make_span(d), [] { return 3; });
    CHECK(cache.version("a") < cache.version("b"));
    CHECK(cache.version("b") < cache.version("c"));

    // Rebuilding 'a' doesn't affect 'b' or 'c'
    auto vb = cache.version("b");
    auto vc = cache.version("c");
    std::vector<uint64_t> d2 = {99};
    cache.get_or_build("a", make_span(d2), [] { return 10; });
    CHECK(cache.version("b") == vb);
    CHECK(cache.version("c") == vc);
    CHECK(cache.version("a") > vc);
}

TEST_CASE("DepTrackedCache - invalidate forces rebuild") {
    DepTrackedCache<std::string, int> cache;
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

    // Next hit with same deps: no rebuild (forced_dirty was cleared)
    cache.get_or_build("a", make_span(d), [&] {
        ++builds;
        return 3;
    });
    CHECK(builds == 2);
}

TEST_CASE("DepTrackedCache - erase drops entry; fresh build after") {
    DepTrackedCache<std::string, int> cache;
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

TEST_CASE("DepTrackedCache - reference stability across unrelated inserts") {
    DepTrackedCache<std::string, int> cache;
    std::vector<uint64_t> d = {};
    const int& ref_a = cache.get_or_build("a", make_span(d), [] { return 111; });
    CHECK(ref_a == 111);
    // Insert many other entries to force rehash.
    for (int i = 0; i < 1024; ++i) {
        cache.get_or_build("k" + std::to_string(i), make_span(d), [i] { return i; });
    }
    // 'a' reference remains valid because Entries are heap-allocated.
    CHECK(ref_a == 111);
    CHECK(&cache.get_or_build("a", make_span(d), [] { return 999; }) == &ref_a);
}

TEST_CASE("DepTrackedCache - empty deps works") {
    DepTrackedCache<std::string, int> cache;
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

TEST_CASE("DepTrackedCache - custom hash/eq with tuple key") {
    using Key = std::tuple<int, int>;
    DepTrackedCache<Key, std::string> cache;
    std::vector<uint64_t> d = {};
    cache.get_or_build(Key{1, 2}, make_span(d), [] { return std::string{"a"}; });
    cache.get_or_build(Key{1, 3}, make_span(d), [] { return std::string{"b"}; });
    CHECK(cache.size() == 2);
    CHECK(*cache.find(Key{1, 2}) == "a");
    CHECK(*cache.find(Key{1, 3}) == "b");
    CHECK(cache.find(Key{9, 9}) == nullptr);
}

TEST_CASE("DepTrackedCache - cascading invalidation: dep version bumps propagate") {
    // Simulate: BGL cache -> pipeline cache -> descriptor cache.
    DepTrackedCache<std::string, int> bgl;   // value: dummy
    DepTrackedCache<std::string, int> pipe;  // pipeline depends on bgl version
    DepTrackedCache<std::string, int> desc;  // descriptor depends on bgl version

    bgl.get_or_build("layout", Span{}, [] { return 1; });
    auto bgl_v1 = bgl.version("layout");

    uint64_t p_deps1[] = {bgl_v1};
    pipe.get_or_build("pl", Span{p_deps1, 1}, [] { return 10; });
    auto pv1 = pipe.version("pl");

    uint64_t d_deps1[] = {bgl_v1, 42 /* some resource version */};
    desc.get_or_build("dg", Span{d_deps1, 2}, [] { return 100; });
    auto dv1 = desc.version("dg");

    // Bump BGL: invalidate then rebuild with same deps (or just erase/recreate).
    bgl.invalidate("layout");
    bgl.get_or_build("layout", Span{}, [] { return 2; });
    auto bgl_v2 = bgl.version("layout");
    CHECK(bgl_v2 > bgl_v1);

    // Pipeline now sees changed dep -> rebuilds
    uint64_t p_deps2[] = {bgl_v2};
    int rebuilds = 0;
    pipe.get_or_build("pl", Span{p_deps2, 1}, [&] {
        ++rebuilds;
        return 11;
    });
    CHECK(rebuilds == 1);
    CHECK(pipe.version("pl") > pv1);

    // Descriptor also sees changed dep -> rebuilds
    uint64_t d_deps2[] = {bgl_v2, 42};
    int desc_rebuilds = 0;
    desc.get_or_build("dg", Span{d_deps2, 2}, [&] {
        ++desc_rebuilds;
        return 101;
    });
    CHECK(desc_rebuilds == 1);
    CHECK(desc.version("dg") > dv1);
}

TEST_CASE("DepTrackedCache - different dep ordering counts as change") {
    DepTrackedCache<std::string, int> cache;
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
