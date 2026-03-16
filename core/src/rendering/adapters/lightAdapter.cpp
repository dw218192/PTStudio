#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/lightAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>

namespace pts::rendering {

LightAdapter& LightAdapter::instance() {
    static LightAdapter s_instance;
    return s_instance;
}

bool LightAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.HasAPI<pxr::UsdLuxLightAPI>();
}

void LightAdapter::sync(pxr::UsdPrim prim, SyncScope& scope, const webgpu::Device& /*device*/) {
    Light light;

    // Common attributes via LightAPI
    pxr::UsdLuxLightAPI light_api(prim);
    pxr::GfVec3f color(1.0f);
    light_api.GetColorAttr().Get(&color);
    light.color = {color[0], color[1], color[2]};

    float intensity = 1.0f;
    light_api.GetIntensityAttr().Get(&intensity);
    light.intensity = intensity;

    // Transform
    light.transform = compute_world_transform(prim);

    // Type-specific attributes
    if (prim.IsA<pxr::UsdLuxDistantLight>()) {
        light.type = Light::Type::Distant;
        // Direction = negative Z axis in light's local space, transformed to world.
        glm::vec4 local_dir(0.0f, 0.0f, -1.0f, 0.0f);
        glm::vec3 world_dir = glm::normalize(glm::vec3(light.transform * local_dir));
        light.direction = world_dir;
    } else if (prim.IsA<pxr::UsdLuxSphereLight>()) {
        light.type = Light::Type::Sphere;
        pxr::UsdLuxSphereLight sphere_light(prim);
        float radius = 0.0f;
        sphere_light.GetRadiusAttr().Get(&radius);
        light.radius = radius;
    } else if (prim.IsA<pxr::UsdLuxRectLight>()) {
        light.type = Light::Type::Rect;
        pxr::UsdLuxRectLight rect_light(prim);
        float w = 1.0f, h = 1.0f;
        rect_light.GetWidthAttr().Get(&w);
        rect_light.GetHeightAttr().Get(&h);
        light.width = w;
        light.height = h;
    } else if (prim.IsA<pxr::UsdLuxDiskLight>()) {
        light.type = Light::Type::Disk;
        pxr::UsdLuxDiskLight disk_light(prim);
        float radius = 0.0f;
        disk_light.GetRadiusAttr().Get(&radius);
        light.radius = radius;
    } else if (prim.IsA<pxr::UsdLuxDomeLight>()) {
        light.type = Light::Type::Dome;
    } else {
        return;
    }

    sync_light(prim, scope, light);
}

std::vector<PropertyDescriptor> LightAdapter::get_properties(const pxr::UsdPrim& prim) const {
    std::vector<PropertyDescriptor> props;

    pxr::UsdLuxLightAPI light_api(prim);

    pxr::GfVec3f color(1.0f);
    light_api.GetColorAttr().Get(&color);
    props.push_back({"inputs:color", "Color", std::any(color), PropertyTag::Color});

    float intensity = 1.0f;
    light_api.GetIntensityAttr().Get(&intensity);
    props.push_back({"inputs:intensity", "Intensity", std::any(intensity)});

    float exposure = 0.0f;
    light_api.GetExposureAttr().Get(&exposure);
    props.push_back({"inputs:exposure", "Exposure", std::any(exposure)});

    if (prim.IsA<pxr::UsdLuxDistantLight>()) {
        pxr::UsdLuxDistantLight distant(prim);
        float angle = 0.53f;
        distant.GetAngleAttr().Get(&angle);
        props.push_back({"inputs:angle", "Angle", std::any(angle)});
    } else if (prim.IsA<pxr::UsdLuxSphereLight>()) {
        pxr::UsdLuxSphereLight sphere(prim);
        float radius = 0.0f;
        sphere.GetRadiusAttr().Get(&radius);
        props.push_back({"inputs:radius", "Radius", std::any(radius)});
    } else if (prim.IsA<pxr::UsdLuxRectLight>()) {
        pxr::UsdLuxRectLight rect(prim);
        float w = 1.0f, h = 1.0f;
        rect.GetWidthAttr().Get(&w);
        rect.GetHeightAttr().Get(&h);
        props.push_back({"inputs:width", "Width", std::any(w)});
        props.push_back({"inputs:height", "Height", std::any(h)});
    } else if (prim.IsA<pxr::UsdLuxDiskLight>()) {
        pxr::UsdLuxDiskLight disk(prim);
        float radius = 0.0f;
        disk.GetRadiusAttr().Get(&radius);
        props.push_back({"inputs:radius", "Radius", std::any(radius)});
    }

    return props;
}

}  // namespace pts::rendering
