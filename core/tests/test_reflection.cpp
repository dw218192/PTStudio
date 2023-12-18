#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "reflection.h"

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
		using R = int const volatile (*const)(int const* volatile* const,
		                                      float const volatile);
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

		auto type = PTS::Type::of<int(int, float, int (*)(void))>();
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

struct Base {
	BEGIN_REFLECT(Base, void);
	FIELD(int, x, 0);

	FIELD(std::vector<int>, y, {});

	FIELD(float, z, 0);

	END_REFLECT();

	~Base() {
		if (m_callback) {
			get_field_info<0>().unregister_on_change_callback(m_callback.value());
		}
	}

	int m_init = [this] {
		get_field_info<0>().register_on_change_callback([this](auto data) {
			if (&data.obj != this) {
				return;
			}
			data.field_info.get(*this) = data.new_val / 2;
		});
		return 0;
	}();
	std::optional<size_t> m_callback;
};

template <typename FieldInfo, typename T>
void set_field(Base& b, FieldInfo&& info, T&& val) {
	auto old_val = val;
	info.get(b) = std::forward<T>(val);
	info.on_change(old_val, info.get(b), b);
}

TEST_CASE("Reflection Macros", "Reflection") {
	SECTION("basic member looping") {
		auto field_names = std::array{
			"x", "y", "z"
		};
		auto field_types = std::array{
			"int", "std::vector<int>", "float"
		};
		Base::for_each_field([&, i = 0](auto field) mutable {
			REQUIRE(field.var_name == field_names[i]);
			REQUIRE(field.type_name == field_types[i]);

			++i;
		});
	}

	SECTION("field info") {
		Base b;
		{
			auto info = Base::get_field_info<0>();
			info.get(b) = 10;
			REQUIRE(b.x == 10);
		}
		{
			auto info = Base::get_field_info<1>();
			info.get(b).push_back(10);
			REQUIRE(b.y[0] == 10);
		}
		{
			auto info = Base::get_field_info<2>();
			info.get(b) = 10;
			REQUIRE(std::abs(b.z - 10.0) < std::numeric_limits<float>::epsilon());
		}
	}

	SECTION("on change callback") {
		auto bs = std::array<Base, 3>{};
		set_field(bs[0], Base::get_field_info<0>(), 100);
		REQUIRE(bs[0].x == 50);
		REQUIRE(bs[1].x == 0);
		REQUIRE(bs[2].x == 0);

		set_field(bs[1], Base::get_field_info<0>(), 200);
		REQUIRE(bs[0].x == 50);
		REQUIRE(bs[1].x == 100);
		REQUIRE(bs[2].x == 0);
	}
}
