#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <arena.h>
#include <vector>
#include <unordered_map>
#include <cstdlib> // For rand()

struct A {
    A() : data{ new int[123] } {}
    virtual ~A() {
        delete[] data;
    }

protected:
    int* data;
};

struct B : A {
    B() : A() {
        data = malloc(100);
    }
    ~B() override {
        free(data);
    }

    void* data;
    std::vector<int> v;
    std::unordered_map<int, int> ump;
};

TEST_CASE("stress alloc and dealloc with random access", "[PTS::FixedSizePoolAllocator]") {
    auto alloc = PTS::FixedSizePoolAllocator{ sizeof(B) };
    struct Deleter {
        Deleter(PTS::Address addr) : addr(addr) {}
        ~Deleter() {
            auto const a = static_cast<A*>(addr.get());
            a->~A();
        	addr.deallocate();
        }
        PTS::Address addr;
    };

    auto allocated = std::vector<Deleter>{};
    // Randomly keep some objects alive to access later
    std::vector<B*> keepAlive;
    for (int i = 0; i < 1000; ++i) {
        auto mem = alloc.allocate();
        auto raw = mem.get();
        auto b = new (raw) B();
        allocated.emplace_back(mem);

        // Randomly decide to keep this object alive for later access
        if (rand() % 3) {
            keepAlive.push_back(b);
        }

        // Randomly deallocate an object
        if (!allocated.empty() && rand() % 5 == 0) {
            allocated.erase(allocated.begin() + (rand() % allocated.size()));
        }
    }

    // Access some data in the kept alive objects to ensure validity
    for (auto b : keepAlive) {
        b->v.push_back(42); // Example access
        REQUIRE(b->v.front() == 42); // Verify access doesn't cause a crash
    }

    // At this point, destructors for B objects will be called automatically for any remaining objects in `allocated`.
    // The `FixedSizePoolAllocator` should ensure no memory leaks occur during deallocation.
}
