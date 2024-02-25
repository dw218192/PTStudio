#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <array>
#include <bitset>
#include <algorithm>

#include <vector>
#include <unordered_map>

#include <cstdlib> // For rand()
#include <arena.h>

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
