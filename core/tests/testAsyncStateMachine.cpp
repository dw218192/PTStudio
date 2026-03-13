#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/webgpu/asyncStateMachine.h>
#include <doctest/doctest.h>

namespace {

// -- test states --------------------------------------------------------------

struct Idle {};

struct Requesting {
    explicit Requesting(int id = 0) : request_id{id} {
    }
    int request_id;
};

struct Ready {
    explicit Ready(int r = 0) : result{r} {
    }
    int result;
};

struct Failed {
    explicit Failed(int e = 0) : error_code{e} {
    }
    int error_code;
};

// -- concrete derived class ---------------------------------------------------

using Base = pts::webgpu::AsyncStateMachine<class FakeMachine, Idle, Requesting, Ready, Failed>;

class FakeMachine : public Base {
   public:
    using Base::get;
    using Base::get_if;
    using Base::is;
    using Base::tick;
    using Base::tick_until_settled;
    using Base::transition;

    FakeMachine() = default;
    FakeMachine(FakeMachine&&) noexcept = default;
    FakeMachine& operator=(FakeMachine&&) noexcept = default;

    int tick_count = 0;

    void on_tick() {
        ++tick_count;
    }

    bool is_pending() const {
        return is<Requesting>();
    }

    WGPUInstance wgpu_instance() const {
        return nullptr;
    }
};

// Auto-settling machine for tick_until_settled test
using AutoBase = pts::webgpu::AsyncStateMachine<class AutoSettleMachine, Idle, Requesting, Ready>;

class AutoSettleMachine : public AutoBase {
   public:
    using AutoBase::get;
    using AutoBase::is;
    using AutoBase::tick_until_settled;
    using AutoBase::transition;

    int ticks_remaining = 3;

    void on_tick() {
        if (--ticks_remaining <= 0) {
            transition<Ready>(99);
        }
    }

    bool is_pending() const {
        return is<Requesting>();
    }
    WGPUInstance wgpu_instance() const {
        return nullptr;
    }
};

}  // namespace

TEST_CASE("AsyncStateMachine - default state is first variant alternative") {
    FakeMachine m;
    CHECK(m.is<Idle>());
    CHECK_FALSE(m.is<Requesting>());
    CHECK_FALSE(m.is<Ready>());
}

TEST_CASE("AsyncStateMachine - transition changes state") {
    FakeMachine m;
    m.transition<Requesting>(42);
    CHECK(m.is<Requesting>());
    CHECK(m.get<Requesting>().request_id == 42);
}

TEST_CASE("AsyncStateMachine - get_if returns nullptr on wrong state") {
    FakeMachine m;
    CHECK(m.get_if<Idle>() != nullptr);
    CHECK(m.get_if<Ready>() == nullptr);
}

TEST_CASE("AsyncStateMachine - tick calls on_tick") {
    FakeMachine m;
    CHECK(m.tick_count == 0);
    m.tick();
    CHECK(m.tick_count == 1);
    m.tick();
    CHECK(m.tick_count == 2);
}

TEST_CASE("AsyncStateMachine - tick_until_settled loops until not pending") {
    AutoSettleMachine am;
    am.transition<Requesting>(1);
    CHECK(am.is_pending());
    am.tick_until_settled();
    CHECK_FALSE(am.is_pending());
    CHECK(am.is<Ready>());
    CHECK(am.get<Ready>().result == 99);
}

TEST_CASE("AsyncStateMachine - move when not pending") {
    FakeMachine m;
    m.transition<Ready>(100);
    CHECK_FALSE(m.is_pending());

    FakeMachine m2{std::move(m)};
    CHECK(m2.is<Ready>());
    CHECK(m2.get<Ready>().result == 100);
}

TEST_CASE("AsyncStateMachine - move assignment when not pending") {
    FakeMachine m;
    m.transition<Ready>(200);

    FakeMachine m2;
    m2 = std::move(m);
    CHECK(m2.is<Ready>());
    CHECK(m2.get<Ready>().result == 200);
}

TEST_CASE("AsyncStateMachine - copy is deleted") {
    CHECK_FALSE(std::is_copy_constructible_v<FakeMachine>);
    CHECK_FALSE(std::is_copy_assignable_v<FakeMachine>);
}
