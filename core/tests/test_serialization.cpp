#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "reflection.h"
#include <array>
#include "jsonArchive.h"
#include "sceneObject.h"
#include "objectRegistry.h"

struct Simple : PTS::SceneObject {
    BEGIN_REFLECT(Simple, void);
    FIELD(int, a, 0);
    FIELD(float, b, 0);
    FIELD(double, c, 0);
    FIELD(char, d, 0);

    using ArrayType5 = std::array<int, 5>;
	FIELD(ArrayType5, e, {});
    END_REFLECT();
};

TEST_CASE("simple", "[serialization]") {
    auto archive = PTS::JsonArchive{};
    auto scene = PTS::Scene{};
    auto cam = PTS::Camera{};
    scene.add_object(Simple{});
    auto res = archive.save(scene, cam);
    REQUIRE(res.has_value());
    INFO("serialized = \n", res.value());
}