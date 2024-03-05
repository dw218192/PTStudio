#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <array>
#include <bitset>
#include <algorithm>

#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <cstdlib> // For rand()

#include <memory/arena.h>
#include <object/object.h>

using Acc = PTS::FixedSizePoolAllocator::Accessor;

namespace {
	auto check_invariant(PTS::FixedSizePoolAllocator& alloc) {
		auto current = Acc::free_list(alloc);
		auto num_free = 0u;
		while (current < Acc::num_initialized(alloc)) {
			++num_free;
			auto const p = Acc::addr_from_index(alloc, current);
			auto const next = *static_cast<size_t*>(p);
			REQUIRE(next != current);
			current = next;
		}

		REQUIRE(current == Acc::num_initialized(alloc));
		REQUIRE(num_free == Acc::num_initialized(alloc) - Acc::num_used(alloc));
	}
}


struct A {
	A() : data{new int[size()]} {}

	A(A const& other) : data{new int[size()]} {
		std::copy(other.data, other.data + size(), data);
	}

	virtual ~A() {
		delete[] data;
	}

	auto set(size_t i, int j) const -> void {
		REQUIRE(i < size());
		data[i] = j;
	}

	auto get(size_t i) const -> int {
		return data[i];
	}

	static auto size() -> size_t {
		return 123u;
	}

protected:
	int* data;
};

struct B final : A {
	std::vector<int> v;
	std::unordered_map<int, int> ump;
};

static constexpr int k_test_size = 1000;
static auto init = []() {
	srand(time(nullptr));
	return 0;
}();

#pragma region PoolAllocatorTest
TEST_CASE("basic alloc and dealloc", "[PTS::FixedSizePoolAllocator]") {
	auto alloc = PTS::FixedSizePoolAllocator{40};
	auto allocated = std::array<PTS::Address, k_test_size>{};
	for (auto i = 0; i < k_test_size; ++i) {
		auto mem = alloc.allocate();
		auto raw = mem.get();

		REQUIRE(raw != nullptr);
		allocated[i] = mem;

		check_invariant(alloc);
	}

	REQUIRE(Acc::num_used(alloc) == k_test_size);
	for (auto addr : allocated) {
		addr.deallocate();
		check_invariant(alloc);
	}
	REQUIRE(Acc::num_used(alloc) == 0);
}

TEST_CASE("interleaved alloc and dealloc", "[PTS::FixedSizePoolAllocator]") {
	auto alloc = PTS::FixedSizePoolAllocator{40};
	auto allocated = std::array<PTS::Address, k_test_size>{};
	auto freed = 0;
	for (auto i = 0; i < k_test_size; ++i) {
		auto mem = alloc.allocate();
		auto raw = mem.get();
		REQUIRE(raw != nullptr);
		allocated[i] = mem;

		if (rand() % 5 == 0) {
			mem = allocated[rand() % (i + 1)];
			if (mem) {
				mem.deallocate();
				++freed;
			}
		}

		check_invariant(alloc);
	}
	REQUIRE(Acc::num_used(alloc) == k_test_size - freed);

	for (auto addr : allocated) {
		// test double-free
		addr.deallocate();
		check_invariant(alloc);
	}
	REQUIRE(Acc::num_used(alloc) == 0);
}

TEST_CASE("basic alloc and dealloc with access", "[PTS::FixedSizePoolAllocator]") {
	auto alloc = PTS::FixedSizePoolAllocator{sizeof(B)};
	auto allocated = std::array<PTS::Address, k_test_size>{};
	for (auto i = 0; i < k_test_size; ++i) {
		auto mem = alloc.allocate();
		auto raw = mem.get();
		REQUIRE(raw != nullptr);
		allocated[i] = mem;

		auto const b = new(raw) B();
		b->set(i % B::size(), i);
		b->v.emplace_back(i);

		check_invariant(alloc);
	}
	REQUIRE(Acc::num_used(alloc) == allocated.size());

	for (auto i = 0; i < k_test_size; ++i) {
		auto const b = static_cast<B*>(allocated[i].get());

		REQUIRE(b->get(i % B::size()) == i);
		REQUIRE(b->v.size() == 1);
		REQUIRE(b->v.front() == i);

		b->~B();
		allocated[i].deallocate();
		check_invariant(alloc);
	}
	REQUIRE(Acc::num_used(alloc) == 0);
}

TEST_CASE("interleaved alloc and dealloc with access", "[PTS::FixedSizePoolAllocator]") {
	auto alloc = PTS::FixedSizePoolAllocator{sizeof(B)};
	auto allocated = std::array<PTS::Address, k_test_size>{};
	auto freed = std::bitset<k_test_size>{};

	for (auto i = 0; i < k_test_size; ++i) {
		auto mem = alloc.allocate();
		REQUIRE(mem.get() != nullptr);
		allocated[i] = mem;

		auto const b = new(mem.get()) B();
		b->set(i % B::size(), i);
		b->v.emplace_back(i);
		b->v.emplace_back(rand());
		b->ump[b->v.back()] = b->v.front();

		if (rand() % 5 == 0) {
			auto j = rand() % (i + 1);
			auto& mem_ref = allocated[j];
			if (mem_ref) {
				static_cast<B*>(mem_ref.get())->~B();
				mem_ref.deallocate();
				freed[j] = true;
			}
			REQUIRE(!mem_ref);
		}

		check_invariant(alloc);
	}
	REQUIRE(Acc::num_used(alloc) == k_test_size - freed.count());

	for (auto i = 0; i < k_test_size; ++i) {
		if (allocated[i]) {
			REQUIRE(!freed[i]);

			auto const b = static_cast<B*>(allocated[i].get());

			REQUIRE(b->get(i % B::size()) == i);
			REQUIRE(b->v.size() == 2);
			REQUIRE(b->v.front() == i);
			REQUIRE(b->ump.count(b->v.back()));
			REQUIRE(b->ump[b->v.back()] == b->v.front());

			b->~B();
			allocated[i].deallocate();
			check_invariant(alloc);
		}
	}
	REQUIRE(Acc::num_used(alloc) == 0);
}

#pragma endregion PoolAllocatorTest


namespace {
	auto ensure_destruct = 0ull;
}

struct TestObj : PTS::Object {
	TestObj(int x, int y, int z) : Object{""}, x(x), y(y), z(z) {
		++ensure_destruct;
	}

	~TestObj() override {
		--ensure_destruct;
	}

	int x, y, z;
};

struct TestObj2 : TestObj {
	TestObj2(int x, int y, int z) : TestObj{x, y, z} {
		save = rand();
		ensure_destruct += save;
	}

	~TestObj2() override {
		ensure_destruct -= save;
	}

	int save;
};

struct TestObj3 : TestObj {
	TestObj3(int x) : TestObj{x, 0, 0} {
		save = ensure_destruct;
		ensure_destruct += ensure_destruct;
	}

	~TestObj3() override {
		ensure_destruct -= save;
	}

	decltype(ensure_destruct) save;
};

TEST_CASE("basic arena alloc and dealloc with access", "[PTS::Arena]") {
	auto& arena = PTS::Arena::get_or_create_arena(0);
	auto objects = std::array<PTS::Handle<PTS::Object>, k_test_size>{};
	auto ids = std::unordered_set<PTS::ObjectID>{};

	for (auto i = 0; i < k_test_size; ++i) {
		auto pobj = PTS::Handle<PTS::Object>{};
		switch (rand() % 3) {
		case 0: {
			pobj = arena.allocate<TestObj>(1, 2, 3);
			REQUIRE(!pobj.as<TestObj2>());
			REQUIRE(!pobj.as<TestObj3>());
			break;
		}
		case 1: {
			auto pobj2 = arena.allocate<TestObj2>(1, 2, 3);
			REQUIRE(pobj2.as<TestObj2>() == pobj2);
			pobj = pobj2;
			REQUIRE(pobj.as<TestObj2>() == pobj2);
			REQUIRE(!pobj.as<TestObj3>());
			break;
		}
		default: {
			auto pobj3 = arena.allocate<TestObj3>(1);
			REQUIRE(pobj3.as<TestObj3>() == pobj3);
			pobj = pobj3;
			REQUIRE(pobj.as<TestObj3>() == pobj3);
			REQUIRE(!pobj.as<TestObj2>());
			break;
		}
		}

		objects[i] = pobj;
		ids.emplace(pobj->get_id());

		REQUIRE(&pobj.get_arena() == &arena);
		REQUIRE(pobj.is_alive());
		REQUIRE(arena.get(pobj->get_id()) == pobj.get());
		REQUIRE(pobj.get() != nullptr);
	}

	REQUIRE(ids.size() == k_test_size);
	for (auto pobj : objects) {
		REQUIRE(pobj.is_alive());
		auto const id = pobj->get_id();

		arena.deallocate(pobj->get_id());
		REQUIRE(!pobj.is_alive());
		REQUIRE(!arena.get(id));
	}

	REQUIRE(!ensure_destruct);
}

TEST_CASE("interleaved arena alloc and dealloc with access", "[PTS::Arena]") {
	auto& arena = PTS::Arena::get_or_create_arena(0);
	auto objects = std::array<PTS::Handle<PTS::Object>, k_test_size>{};
	auto ids = std::unordered_set<PTS::ObjectID>{};
	auto freed = 0;

	for (auto i = 0; i < k_test_size; ++i) {
		auto pobj = PTS::Handle<PTS::Object>{};
		switch (rand() % 3) {
		case 0: {
			pobj = arena.allocate<TestObj>(1, 2, 3);
			REQUIRE(!pobj.as<TestObj2>());
			REQUIRE(!pobj.as<TestObj3>());
			break;
		}
		case 1: {
			auto pobj2 = arena.allocate<TestObj2>(1, 2, 3);
			REQUIRE(pobj2.as<TestObj2>() == pobj2);
			pobj = pobj2;
			REQUIRE(pobj.as<TestObj2>() == pobj2);
			REQUIRE(!pobj.as<TestObj3>());
			break;
		}
		default: {
			auto pobj3 = arena.allocate<TestObj3>(1);
			REQUIRE(pobj3.as<TestObj3>() == pobj3);
			pobj = pobj3;
			REQUIRE(pobj.as<TestObj3>() == pobj3);
			REQUIRE(!pobj.as<TestObj2>());
			break;
		}
		}

		objects[i] = pobj;
		ids.emplace(pobj->get_id());

		REQUIRE(&pobj.get_arena() == &arena);
		REQUIRE(pobj.is_alive());
		REQUIRE(arena.get(pobj->get_id()) == pobj.get());
		REQUIRE(pobj.get() != nullptr);

		if (rand() % 5 == 0) {
			auto j = rand() % (i + 1);
			if (objects[j].is_alive()) {
				ids.erase(objects[j]->get_id());
				arena.deallocate(objects[j]->get_id());
				REQUIRE(!objects[j].is_alive());
				++freed;
			}
		}
	}

	REQUIRE(ids.size() == k_test_size - freed);
	for (auto pobj : objects) {
		if (pobj.is_alive()) {
			arena.deallocate(pobj->get_id());
			REQUIRE(!pobj.is_alive());
		}
	}

	REQUIRE(!ensure_destruct);
}
