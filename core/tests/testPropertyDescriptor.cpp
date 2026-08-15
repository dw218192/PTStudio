#include <core/rendering/adapters/capsuleAdapter.h>
#include <core/rendering/adapters/coneAdapter.h>
#include <core/rendering/adapters/cubeAdapter.h>
#include <core/rendering/adapters/cylinderAdapter.h>
#include <core/rendering/adapters/lightAdapter.h>
#include <core/rendering/adapters/meshAdapter.h>
#include <core/rendering/adapters/sphereAdapter.h>
#include <core/rendering/propertyDescriptor.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>

#include "testApplication.h"

namespace {

template <typename T>
T prop_value(const std::vector<pts::rendering::PropertyDescriptor>& props,
             const std::string& name) {
    for (const auto& p : props) {
        if (p.name == name) return std::any_cast<T>(p.value);
    }
    FAIL("property '" << name << "' not found");
    return T{};
}

bool has_prop(const std::vector<pts::rendering::PropertyDescriptor>& props,
              const std::string& name) {
    for (const auto& p : props) {
        if (p.name == name) return true;
    }
    return false;
}

pts::rendering::PropertyTag prop_tags(const std::vector<pts::rendering::PropertyDescriptor>& props,
                                      const std::string& name) {
    for (const auto& p : props) {
        if (p.name == name) return p.tags;
    }
    FAIL("property '" << name << "' not found");
    return pts::rendering::PropertyTag::None;
}

}  // namespace

TEST_CASE("PropertyTag bitwise operators") {
    using pts::rendering::PropertyTag;
    auto combined = PropertyTag::Color | PropertyTag::ReadOnly;
    CHECK(pts::rendering::has_tag(combined, PropertyTag::Color));
    CHECK(pts::rendering::has_tag(combined, PropertyTag::ReadOnly));
    CHECK(!pts::rendering::has_tag(PropertyTag::None, PropertyTag::Color));
    CHECK(!pts::rendering::has_tag(PropertyTag::Color, PropertyTag::ReadOnly));
}

TEST_CASE("CubeAdapter::get_properties returns size") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cube = pxr::UsdGeomCube::Define(stage, pxr::SdfPath("/Cube"));
    cube.GetSizeAttr().Set(4.0);

    auto props = pts::rendering::CubeAdapter::instance().get_properties(cube.GetPrim());
    REQUIRE(props.size() == 1);
    CHECK(props[0].name == "size");
    CHECK(props[0].label == "Size");
    CHECK(std::any_cast<double>(props[0].value) == doctest::Approx(4.0));
}

TEST_CASE("SphereAdapter::get_properties returns radius") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto sphere = pxr::UsdGeomSphere::Define(stage, pxr::SdfPath("/Sphere"));
    sphere.GetRadiusAttr().Set(3.0);

    auto props = pts::rendering::SphereAdapter::instance().get_properties(sphere.GetPrim());
    REQUIRE(props.size() == 1);
    CHECK(props[0].name == "radius");
    CHECK(std::any_cast<double>(props[0].value) == doctest::Approx(3.0));
}

TEST_CASE("CylinderAdapter::get_properties returns radius and height") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cyl = pxr::UsdGeomCylinder::Define(stage, pxr::SdfPath("/Cyl"));
    cyl.GetRadiusAttr().Set(2.0);
    cyl.GetHeightAttr().Set(5.0);

    auto props = pts::rendering::CylinderAdapter::instance().get_properties(cyl.GetPrim());
    REQUIRE(props.size() == 2);
    CHECK(prop_value<double>(props, "radius") == doctest::Approx(2.0));
    CHECK(prop_value<double>(props, "height") == doctest::Approx(5.0));
}

TEST_CASE("ConeAdapter::get_properties returns radius and height") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cone = pxr::UsdGeomCone::Define(stage, pxr::SdfPath("/Cone"));
    cone.GetRadiusAttr().Set(1.5);
    cone.GetHeightAttr().Set(3.0);

    auto props = pts::rendering::ConeAdapter::instance().get_properties(cone.GetPrim());
    REQUIRE(props.size() == 2);
    CHECK(prop_value<double>(props, "radius") == doctest::Approx(1.5));
    CHECK(prop_value<double>(props, "height") == doctest::Approx(3.0));
}

TEST_CASE("CapsuleAdapter::get_properties returns radius and height") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto cap = pxr::UsdGeomCapsule::Define(stage, pxr::SdfPath("/Capsule"));
    cap.GetRadiusAttr().Set(0.75);
    cap.GetHeightAttr().Set(2.0);

    auto props = pts::rendering::CapsuleAdapter::instance().get_properties(cap.GetPrim());
    REQUIRE(props.size() == 2);
    CHECK(prop_value<double>(props, "radius") == doctest::Approx(0.75));
    CHECK(prop_value<double>(props, "height") == doctest::Approx(2.0));
}

TEST_CASE("MeshAdapter::get_properties returns empty (default)") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto mesh = pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/Mesh"));

    auto props = pts::rendering::MeshAdapter::instance().get_properties(mesh.GetPrim());
    CHECK(props.empty());
}

TEST_CASE("LightAdapter::get_properties - sphere light") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto light = pxr::UsdLuxSphereLight::Define(stage, pxr::SdfPath("/Light"));
    light.GetColorAttr().Set(pxr::GfVec3f(1.0f, 0.5f, 0.0f));
    light.GetIntensityAttr().Set(2.5f);
    light.GetExposureAttr().Set(1.0f);
    light.GetRadiusAttr().Set(0.5f);

    auto props = pts::rendering::LightAdapter::instance().get_properties(light.GetPrim());
    REQUIRE(props.size() >= 4);

    auto color = prop_value<pxr::GfVec3f>(props, "inputs:color");
    CHECK(color[0] == doctest::Approx(1.0f));
    CHECK(color[1] == doctest::Approx(0.5f));
    CHECK(color[2] == doctest::Approx(0.0f));
    CHECK(pts::rendering::has_tag(prop_tags(props, "inputs:color"),
                                  pts::rendering::PropertyTag::Color));

    CHECK(prop_value<float>(props, "inputs:intensity") == doctest::Approx(2.5f));
    CHECK(prop_value<float>(props, "inputs:exposure") == doctest::Approx(1.0f));
    CHECK(prop_value<float>(props, "inputs:radius") == doctest::Approx(0.5f));
}

TEST_CASE("LightAdapter::get_properties - distant light has angle") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto light = pxr::UsdLuxDistantLight::Define(stage, pxr::SdfPath("/DistantLight"));
    light.GetAngleAttr().Set(1.5f);

    auto props = pts::rendering::LightAdapter::instance().get_properties(light.GetPrim());
    CHECK(has_prop(props, "inputs:angle"));
    CHECK(prop_value<float>(props, "inputs:angle") == doctest::Approx(1.5f));
}

TEST_CASE("LightAdapter::get_properties - rect light has width and height") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto light = pxr::UsdLuxRectLight::Define(stage, pxr::SdfPath("/RectLight"));
    light.GetWidthAttr().Set(3.0f);
    light.GetHeightAttr().Set(4.0f);

    auto props = pts::rendering::LightAdapter::instance().get_properties(light.GetPrim());
    CHECK(prop_value<float>(props, "inputs:width") == doctest::Approx(3.0f));
    CHECK(prop_value<float>(props, "inputs:height") == doctest::Approx(4.0f));
}

TEST_CASE("LightAdapter::get_properties - disk light has radius") {
    auto stage = pxr::UsdStage::CreateInMemory();
    auto light = pxr::UsdLuxDiskLight::Define(stage, pxr::SdfPath("/DiskLight"));
    light.GetRadiusAttr().Set(2.0f);

    auto props = pts::rendering::LightAdapter::instance().get_properties(light.GetPrim());
    CHECK(prop_value<float>(props, "inputs:radius") == doctest::Approx(2.0f));
}

PTS_TEST_MAIN()
