#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "reflection.h"

static_assert(
    std::is_same_v<PTS::Traits::get_template_args_t<std::tuple<int, float>>,
                   std::tuple<int, float>>);
static_assert(
    std::is_same_v<PTS::Traits::get_template_args_t<std::pair<int, float>>,
                   std::tuple<int, float>>);
static_assert(std::is_same_v<PTS::Traits::get_template_args_t<std::vector<int>>,
                             std::tuple<int, std::allocator<int>>>);

static_assert(std::is_same_v<
              PTS::Traits::get_template_args_t<std::function<int(int, float)>>,
              std::tuple<int(int, float)>>);

template<typename T>
void f();

TEST_CASE("PTS::Type", "[reflection]") {
    using namespace PTS;
    SECTION("basic test") {
        // basic types, cv qualifiers are ignored
        INFO("testing basic types");
        REQUIRE(PTS::Type::of<int>() == PTS::Type::of<int>());
        REQUIRE(PTS::Type::of<int const>() == PTS::Type::of<int const>());
        REQUIRE(PTS::Type::of<int volatile>() == PTS::Type::of<int>());
        REQUIRE(PTS::Type::of<int>() == PTS::Type::of<int const volatile>());
        REQUIRE(PTS::Type::of<int const volatile>() ==
                PTS::Type::of<int const volatile>());
        REQUIRE(PTS::Type::of<int* const** volatile***>() ==
                PTS::Type::of<int******>());
        REQUIRE(
            PTS::Type::of<int* const* const* volatile** const* volatile>() ==
            PTS::Type::of<int* const** volatile***>());

        // array types are not decayed; cv qualifiers are ignored for the element type
        REQUIRE(PTS::Type::of<int[5]>() != PTS::Type::of<int[51]>());
        REQUIRE(PTS::Type::of<int const[5]>() == PTS::Type::of<int[5]>());


        // array pointer, cv qualifiers are ignored for the pointer itself
        REQUIRE(PTS::Type::of<int(*)[5]>() ==
                PTS::Type::of<int(*const volatile)[5]>());

        using A = int[5];
        REQUIRE(PTS::Type::of<A******>() ==
                PTS::Type::of<A* const** volatile***>());

        // function types, cv qualifiers are ignored for parameters and return
        // type but the function itself is not decayed
        REQUIRE(PTS::Type::of<int const volatile(int const volatile,
                                                 float const* volatile**)>() ==
                PTS::Type::of<int(int, float***)>());

        // function pointers, cv qualifiers are ignored for parameters and for
        // the pointer itself
        REQUIRE(PTS::Type::of<int volatile const (*const)(
                    int const* volatile* const, float volatile const)>() ==
                PTS::Type::of<int (*)(int**, float)>());
        using R = int volatile const (*const)(int const* volatile* const,
                                              float volatile const);
        REQUIRE(PTS::Type::of<R(int, R, R* const* volatile)>() ==
                PTS::Type::of<R(int, R, R**)>());

        // templated types, default template arguments are preserved
        REQUIRE(PTS::Type::of<std::vector<int>>() ==
                PTS::Type::of<std::vector<int, std::allocator<int>>>());

        // reference types, lvalue reference is treated as pointer
        // rvalue reference is treated as value
        REQUIRE(PTS::Type::of<int&>() == PTS::Type::of<int*>());
        REQUIRE(PTS::Type::of<int const&>() == PTS::Type::of<int*>());
        REQUIRE(PTS::Type::of<int volatile&>() == PTS::Type::of<int*>());
        REQUIRE(PTS::Type::of<int&&>() == PTS::Type::of<int>());
        REQUIRE(PTS::Type::of<int const&&>() == PTS::Type::of<int>());
        REQUIRE(PTS::Type::of<int(&)[2]>() == PTS::Type::of<int volatile(&)[2]>());
        REQUIRE(PTS::Type::of<int(&)[10]>() == PTS::Type::of<int const(&)[10]>());
        REQUIRE(PTS::Type::of<int(&&)[10]>() == PTS::Type::of<int[10]>());
        REQUIRE(PTS::Type::of<int const (&)(int)>() == PTS::Type::of<int(*)(int)>());
        REQUIRE(PTS::Type::of<int const (&&)(int)>() ==
                PTS::Type::of<int(int)>());
    }

    SECTION("type trait") {
        REQUIRE(PTS::Type::of<int>().is_arithmetic);
        REQUIRE(PTS::Type::of<float>().is_arithmetic);
        REQUIRE(PTS::Type::of<double>().is_arithmetic);
        REQUIRE(PTS::Type::of<long double>().is_arithmetic);
        REQUIRE(PTS::Type::of<int[5]>().is_container);
        REQUIRE(PTS::Type::of<std::vector<int>>().is_container);
        
        // tuple and pair are not considered containers
        // because they are heterogeneous collections
        REQUIRE(!PTS::Type::of<std::tuple<int, float>>().is_container);
        REQUIRE(!PTS::Type::of<std::pair<int, float>>().is_container);

        REQUIRE(PTS::Type::of<std::variant<int, float>>().is_templated);
        REQUIRE(PTS::Type::of<std::tuple<int, float>>().is_templated);
        REQUIRE(PTS::Type::of<std::pair<int, float>>().is_templated);
        REQUIRE(PTS::Type::of<std::vector<int>>().is_templated);
        REQUIRE(PTS::Type::of<std::function<int(int, float)>>().is_templated);
    }

    SECTION("subtypes") {
        REQUIRE(PTS::Type::of<int*>().pointed_to_type() ==
                PTS::Type::of<int>());
        REQUIRE(PTS::Type::of<float**>().pointed_to_type() ==
                PTS::Type::of<float*>());
        REQUIRE(PTS::Type::of<int******>()
                    .pointed_to_type()
                    .pointed_to_type()
                    .pointed_to_type()
                    .pointed_to_type()
                    .pointed_to_type()
                    .pointed_to_type() == PTS::Type::of<int>());
    
        REQUIRE(PTS::Type::of<int[5]>().contained_type() == PTS::Type::of<int>());
        REQUIRE(PTS::Type::of<int[5][10]>().contained_type() ==
                PTS::Type::of<int[10]>());

        REQUIRE(PTS::Type::of<int(*)[5]>().pointed_to_type() ==
                PTS::Type::of<int[5]>());
        
        auto type = PTS::Type::of<int(int,float,int(*)(void))>();
        REQUIRE(type.func_return_type() == PTS::Type::of<int>());
        REQUIRE(type.func_arg_types() ==
                std::vector{PTS::Type::of<int>(), PTS::Type::of<float>(),
                            PTS::Type::of<int(*)(void)>()});
    }

    SECTION("templated type") {
        auto type = PTS::Type::of<std::tuple<int, float>>();
        auto template_args = type.template_args();
        REQUIRE_THAT(template_args,
                     Catch::Matchers::UnorderedEquals(std::vector{
                         PTS::Type::of<int>(), PTS::Type::of<float>()}));
        REQUIRE(type.to_string() == "std::tuple<int, float>");

        type = PTS::Type::of<std::vector<int>>();
        template_args = type.template_args();
        REQUIRE_THAT(
            template_args,
            Catch::Matchers::UnorderedEquals(std::vector{
                PTS::Type::of<int>(), PTS::Type::of<std::allocator<int>>()}));
        REQUIRE(type.to_string() == "std::vector<int, std::allocator<int>>");

        type = PTS::Type::of<std::pair<
            std::pair<int, std::pair<std::string, float>>, std::vector<int>>>();
        template_args = type.template_args();
        REQUIRE_THAT(
            template_args,
            Catch::Matchers::UnorderedEquals(std::vector{
                PTS::Type::of<std::pair<int, std::pair<std::string, float>>>(),
                PTS::Type::of<std::vector<int>>()}));
        REQUIRE(type.to_string() ==
                "std::pair<std::pair<int, std::pair<std::string<char, "
                "std::char_traits<char>, std::allocator<char>>, float>>, "
                "std::vector<int, std::allocator<int>>>");

        type = PTS::Type::of<std::function<int(int, float)>>();
        template_args = type.template_args();
        REQUIRE_THAT(template_args,
                     Catch::Matchers::UnorderedEquals(std::vector{
                         PTS::Type::of<int(int, float)>()}));
        REQUIRE(type.to_string() == "std::function<int(int, float)>");
    }
}