#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/inPlacePimpl.h>
#include <doctest/doctest.h>

#include <type_traits>
#include <vector>

namespace {

struct MockImpl {
    static int s_destructor_count;

    int counter = 7;
    std::vector<int> data{1, 2, 3};

    MockImpl() = default;
    MockImpl(int c, std::vector<int> d) : counter(c), data(std::move(d)) {
    }
    ~MockImpl() {
        ++s_destructor_count;
    }

    MockImpl(const MockImpl&) = delete;
    MockImpl& operator=(const MockImpl&) = delete;
    MockImpl(MockImpl&&) = default;
    MockImpl& operator=(MockImpl&&) = default;
};

int MockImpl::s_destructor_count = 0;

class MockOwner final : public pts::InPlacePimpl<MockOwner, MockImpl, 128, 16> {
   public:
    MockOwner();
    MockOwner(int c, std::vector<int> d);
    ~MockOwner();

    MockOwner(const MockOwner&) = delete;
    MockOwner& operator=(const MockOwner&) = delete;

    MockOwner(MockOwner&& other) noexcept;
    MockOwner& operator=(MockOwner&& other) noexcept;

    int counter() const {
        return impl().counter;
    }
    const std::vector<int>& data() const {
        return impl().data;
    }
};

MockOwner::MockOwner() {
    construct();
}
MockOwner::MockOwner(int c, std::vector<int> d) {
    construct(c, std::move(d));
}
MockOwner::~MockOwner() {
    destroy();
}

MockOwner::MockOwner(MockOwner&& other) noexcept {
    construct(std::move(other.impl()));
}

MockOwner& MockOwner::operator=(MockOwner&& other) noexcept {
    if (this != &other) {
        destroy();
        construct(std::move(other.impl()));
    }
    return *this;
}

}  // namespace

TEST_CASE("InPlacePimpl - default construct leaves Impl alive") {
    MockImpl::s_destructor_count = 0;
    {
        MockOwner owner;
        CHECK(owner.counter() == 7);
        CHECK(owner.data() == std::vector<int>{1, 2, 3});
        CHECK(MockImpl::s_destructor_count == 0);
    }
    CHECK(MockImpl::s_destructor_count == 1);
}

TEST_CASE("InPlacePimpl - forwarded-args construct") {
    MockImpl::s_destructor_count = 0;
    {
        MockOwner owner(42, std::vector<int>{9, 8});
        CHECK(owner.counter() == 42);
        CHECK(owner.data() == std::vector<int>{9, 8});
    }
    CHECK(MockImpl::s_destructor_count == 1);
}

TEST_CASE("InPlacePimpl - move ctor transfers Impl state") {
    MockImpl::s_destructor_count = 0;
    {
        MockOwner src(11, std::vector<int>{5, 6, 7, 8});
        MockOwner dst(std::move(src));

        CHECK(dst.counter() == 11);
        CHECK(dst.data() == std::vector<int>{5, 6, 7, 8});
        // src's vector was moved-from; counter was copied.
        CHECK(src.counter() == 11);
        CHECK(src.data().empty());
        CHECK(MockImpl::s_destructor_count == 0);
    }
    // Both outer objects live to scope exit, both Impls get destroyed.
    CHECK(MockImpl::s_destructor_count == 2);
}

TEST_CASE("InPlacePimpl - move assign destroys old Impl and constructs new") {
    MockImpl::s_destructor_count = 0;
    {
        MockOwner a(1, std::vector<int>{1});
        MockOwner b(2, std::vector<int>{2, 2});

        b = std::move(a);
        // Old b's Impl was destroyed; new one move-constructed from a.
        CHECK(MockImpl::s_destructor_count == 1);
        CHECK(b.counter() == 1);
        CHECK(b.data() == std::vector<int>{1});
    }
    // a and b both destroyed at scope exit.
    CHECK(MockImpl::s_destructor_count == 3);
}

TEST_CASE("InPlacePimpl - self move-assign is safe") {
    MockImpl::s_destructor_count = 0;
    {
        MockOwner owner(5, std::vector<int>{5});
        MockOwner& ref = owner;
        owner = std::move(ref);
        CHECK(owner.counter() == 5);
        CHECK(MockImpl::s_destructor_count == 0);
    }
    CHECK(MockImpl::s_destructor_count == 1);
}

static_assert(!std::is_copy_constructible_v<MockOwner>, "MockOwner must be non-copyable");
static_assert(!std::is_copy_assignable_v<MockOwner>, "MockOwner must be non-copy-assignable");
static_assert(std::is_move_constructible_v<MockOwner>, "MockOwner must be movable");
static_assert(std::is_move_assignable_v<MockOwner>, "MockOwner must be move-assignable");
