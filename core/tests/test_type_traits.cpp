#include <array>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "typeTraitsUtil.h"

// ----------------- is_container -----------------
static_assert(PTS::Traits::is_container_v<std::vector<int>>);
static_assert(PTS::Traits::is_container_v<std::deque<int>>);
static_assert(PTS::Traits::is_container_v<std::list<int>>);
static_assert(PTS::Traits::is_container_v<std::set<int>>);
static_assert(PTS::Traits::is_container_v<std::map<int, float>>);
static_assert(PTS::Traits::is_container_v<std::unordered_set<int>>);
static_assert(PTS::Traits::is_container_v<std::unordered_map<int, float>>);
static_assert(PTS::Traits::is_container_v<std::string>);
static_assert(PTS::Traits::is_container_v<std::wstring>);
static_assert(PTS::Traits::is_container_v<std::u16string>);
static_assert(PTS::Traits::is_container_v<std::u32string>);
static_assert(PTS::Traits::is_container_v<std::array<int, 5>>);
// pair and tuple are not treated as containers
static_assert(!PTS::Traits::is_container_v<std::tuple<int, float>>);
static_assert(!PTS::Traits::is_container_v<std::pair<int, float>>);
static_assert(!PTS::Traits::is_container_v<std::pair<std::vector<int>, std::vector<float>>>);

static_assert(!PTS::Traits::is_container_v<std::function<int(int, float)>>);
static_assert(!PTS::Traits::is_container_v<int>);
static_assert(!PTS::Traits::is_container_v<float>);
static_assert(!PTS::Traits::is_container_v<double>);
static_assert(!PTS::Traits::is_container_v<bool>);
static_assert(!PTS::Traits::is_container_v<char>);

static_assert(PTS::Traits::is_associative_container_v<std::set<int>>);
static_assert(PTS::Traits::is_associative_container_v<std::map<int, float>>);
static_assert(PTS::Traits::is_associative_container_v<std::unordered_set<int>>);
static_assert(PTS::Traits::is_associative_container_v<std::unordered_map<int, float>>);
static_assert(!PTS::Traits::is_associative_container_v<std::vector<int>>);
static_assert(!PTS::Traits::is_associative_container_v<std::deque<int>>);
static_assert(!PTS::Traits::is_associative_container_v<std::list<int>>);
static_assert(!PTS::Traits::is_associative_container_v<std::string>);
static_assert(!PTS::Traits::is_associative_container_v<std::wstring>);
static_assert(!PTS::Traits::is_associative_container_v<std::u16string>);
static_assert(!PTS::Traits::is_associative_container_v<std::u32string>);
static_assert(!PTS::Traits::is_associative_container_v<std::array<int, 5>>);
static_assert(!PTS::Traits::is_associative_container_v<std::tuple<int, float>>);
static_assert(!PTS::Traits::is_associative_container_v<std::pair<int, float>>);
static_assert(
    !PTS::Traits::is_associative_container_v<std::pair<std::vector<int>, std::vector<float>>>);
static_assert(!PTS::Traits::is_associative_container_v<std::function<int(int, float)>>);
static_assert(!PTS::Traits::is_associative_container_v<int>);
static_assert(!PTS::Traits::is_associative_container_v<float>);
static_assert(!PTS::Traits::is_associative_container_v<double>);
static_assert(!PTS::Traits::is_associative_container_v<bool>);
static_assert(!PTS::Traits::is_associative_container_v<char>);

static_assert(PTS::Traits::is_sequence_container_v<std::vector<int>>);
static_assert(PTS::Traits::is_sequence_container_v<std::deque<int>>);
static_assert(PTS::Traits::is_sequence_container_v<std::list<int>>);
static_assert(PTS::Traits::is_sequence_container_v<std::string>);
static_assert(PTS::Traits::is_sequence_container_v<std::wstring>);
static_assert(PTS::Traits::is_sequence_container_v<std::u16string>);
static_assert(PTS::Traits::is_sequence_container_v<std::u32string>);
static_assert(!PTS::Traits::is_sequence_container_v<std::set<int>>);
static_assert(!PTS::Traits::is_sequence_container_v<std::map<int, float>>);
static_assert(!PTS::Traits::is_sequence_container_v<std::unordered_set<int>>);
static_assert(!PTS::Traits::is_sequence_container_v<std::unordered_map<int, float>>);
static_assert(!PTS::Traits::is_sequence_container_v<std::array<int, 5>>);
static_assert(!PTS::Traits::is_sequence_container_v<std::tuple<int, float>>);
static_assert(!PTS::Traits::is_sequence_container_v<std::pair<int, float>>);
static_assert(
    !PTS::Traits::is_sequence_container_v<std::pair<std::vector<int>, std::vector<float>>>);
static_assert(!PTS::Traits::is_sequence_container_v<std::function<int(int, float)>>);
static_assert(!PTS::Traits::is_sequence_container_v<int>);
static_assert(!PTS::Traits::is_sequence_container_v<float>);
static_assert(!PTS::Traits::is_sequence_container_v<double>);
static_assert(!PTS::Traits::is_sequence_container_v<bool>);
static_assert(!PTS::Traits::is_sequence_container_v<char>);

// ----------------- comparability -----------------
static_assert(PTS::Traits::is_equitable_v<int, int>);
static_assert(PTS::Traits::is_equitable_v<int, float>);
static_assert(PTS::Traits::is_equitable_v<float, int>);
static_assert(PTS::Traits::is_equitable_v<float, float>);
static_assert(PTS::Traits::is_equitable_v<std::string, std::string>);
static_assert(!PTS::Traits::is_equitable_v<std::string, int>);
static_assert(!PTS::Traits::is_equitable_v<std::string, float>);
static_assert(!PTS::Traits::is_equitable_v<std::string, std::vector<int>>);

static_assert(PTS::Traits::is_less_than_comparable_v<int, int>);
static_assert(PTS::Traits::is_less_than_comparable_v<int, float>);
static_assert(PTS::Traits::is_less_than_comparable_v<float, int>);
static_assert(PTS::Traits::is_less_than_comparable_v<float, float>);
static_assert(PTS::Traits::is_less_than_comparable_v<std::string, std::string>);
static_assert(!PTS::Traits::is_less_than_comparable_v<std::string, int>);
static_assert(!PTS::Traits::is_less_than_comparable_v<std::string, float>);
static_assert(!PTS::Traits::is_less_than_comparable_v<std::string, std::vector<int>>);

static_assert(PTS::Traits::is_greater_than_comparable_v<int, int>);
static_assert(PTS::Traits::is_greater_than_comparable_v<int, float>);
static_assert(PTS::Traits::is_greater_than_comparable_v<float, int>);
static_assert(PTS::Traits::is_greater_than_comparable_v<float, float>);
static_assert(PTS::Traits::is_greater_than_comparable_v<std::string, std::string>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<std::string, int>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<std::string, float>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<std::string, std::vector<int>>);

// custom types
struct A {
    int a;
    float b;
    bool operator==(A const& other) const noexcept {
        return a == other.a && b == other.b;
    }
    bool operator<(A const& other) const noexcept {
        return a < other.a && b < other.b;
    }
    bool operator>(A const& other) const noexcept {
        return a > other.a && b > other.b;
    }
};

struct B {
    int a;
    float b;
    bool operator<(B const& other) const noexcept {
        return a < other.a && b < other.b;
    }
    bool operator>(B const& other) const noexcept {
        return a > other.a && b > other.b;
    }
    bool operator==(A const& other) const noexcept {
        return a == other.a && b == other.b;
    }
    bool operator<(A const& other) const noexcept {
        return a < other.a && b < other.b;
    }
    bool operator>(A const& other) const noexcept {
        return a > other.a && b > other.b;
    }
};

static_assert(PTS::Traits::is_equitable_v<A, A>);
static_assert(!PTS::Traits::is_equitable_v<A, B>);
static_assert(PTS::Traits::is_equitable_v<B, A>);
static_assert(!PTS::Traits::is_equitable_v<B, B>);
static_assert(!PTS::Traits::is_equitable_v<A, int>);
static_assert(!PTS::Traits::is_equitable_v<A, float>);
static_assert(!PTS::Traits::is_equitable_v<A, std::string>);
static_assert(!PTS::Traits::is_equitable_v<A, std::vector<int>>);

static_assert(PTS::Traits::is_less_than_comparable_v<A, A>);
static_assert(!PTS::Traits::is_less_than_comparable_v<A, B>);
static_assert(PTS::Traits::is_less_than_comparable_v<B, A>);
static_assert(PTS::Traits::is_less_than_comparable_v<B, B>);
static_assert(!PTS::Traits::is_less_than_comparable_v<A, int>);
static_assert(!PTS::Traits::is_less_than_comparable_v<A, float>);
static_assert(!PTS::Traits::is_less_than_comparable_v<A, std::string>);

static_assert(PTS::Traits::is_greater_than_comparable_v<A, A>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<A, B>);
static_assert(PTS::Traits::is_greater_than_comparable_v<B, A>);
static_assert(PTS::Traits::is_greater_than_comparable_v<B, B>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<A, int>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<A, float>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<A, std::string>);
static_assert(!PTS::Traits::is_greater_than_comparable_v<A, std::vector<int>>);

// ----------------- get_template_args -----------------
static_assert(std::is_same_v<PTS::Traits::get_template_args_t<std::tuple<int, float>>,
                             std::tuple<int, float>>);
static_assert(std::is_same_v<PTS::Traits::get_template_args_t<std::pair<int, float>>,
                             std::tuple<int, float>>);
static_assert(std::is_same_v<PTS::Traits::get_template_args_t<std::vector<int>>,
                             std::tuple<int, std::allocator<int>>>);
static_assert(std::is_same_v<PTS::Traits::get_template_args_t<std::function<int(int, float)>>,
                             std::tuple<int(int, float)>>);

int main() {
    return 0;
}