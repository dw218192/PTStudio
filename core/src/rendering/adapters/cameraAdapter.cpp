#include <core/rendering/adapterHelpers.h>
#include <core/rendering/adapters/cameraAdapter.h>
#include <core/rendering/renderWorld.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <cmath>

namespace pts::rendering {

CameraAdapter& CameraAdapter::instance() {
    static CameraAdapter s_instance;
    return s_instance;
}

bool CameraAdapter::can_adapt(const pxr::UsdPrim& prim) const {
    return prim.IsA<pxr::UsdGeomCamera>();
}

void CameraAdapter::sync(pxr::UsdPrim prim, SyncScope& scope) {
    pxr::UsdGeomCamera cam(prim);

    float focal_length = 50.0f;
    cam.GetFocalLengthAttr().Get(&focal_length);

    float h_aperture = 36.0f;
    cam.GetHorizontalApertureAttr().Get(&h_aperture);

    float v_aperture = 24.0f;
    cam.GetVerticalApertureAttr().Get(&v_aperture);

    pxr::GfVec2f clip_range(0.1f, 10000.0f);
    cam.GetClippingRangeAttr().Get(&clip_range);

    pxr::TfToken projection;
    cam.GetProjectionAttr().Get(&projection);

    auto world_xf = compute_world_transform(prim);

    CameraData data;
    data.view_matrix = glm::inverse(world_xf);
    data.orthographic = (projection == pxr::UsdGeomTokens->orthographic);
    data.fov_y_radians = 2.0f * std::atan(v_aperture / (2.0f * focal_length));
    // USD aperture is in mm; orthographic size is aperture / 10 (cm to scene units)
    data.ortho_height = v_aperture / 10.0f;
    data.near_clip = clip_range[0];
    data.far_clip = clip_range[1];

    sync_camera(prim, scope, data);
}

std::vector<PropertyDescriptor> CameraAdapter::get_properties(const pxr::UsdPrim& prim) const {
    std::vector<PropertyDescriptor> props;
    pxr::UsdGeomCamera cam(prim);

    float focal_length = 50.0f;
    cam.GetFocalLengthAttr().Get(&focal_length);
    props.push_back({"focalLength", "Focal Length", std::any(focal_length), {}, 1.0f, 1.0f});

    float h_aperture = 36.0f;
    cam.GetHorizontalApertureAttr().Get(&h_aperture);
    props.push_back({"horizontalAperture", "H Aperture", std::any(h_aperture), {}, 0.1f, 0.1f});

    float v_aperture = 24.0f;
    cam.GetVerticalApertureAttr().Get(&v_aperture);
    props.push_back({"verticalAperture", "V Aperture", std::any(v_aperture), {}, 0.1f, 0.1f});

    pxr::GfVec2f clip_range(0.1f, 10000.0f);
    cam.GetClippingRangeAttr().Get(&clip_range);
    props.push_back({"clippingRange:0", "Near Clip", std::any(clip_range[0]), {}, 0.01f, 0.001f});
    props.push_back({"clippingRange:1", "Far Clip", std::any(clip_range[1]), {}, 10.0f, 1.0f});

    return props;
}

static pxr::UsdPrim define_camera(const pxr::UsdStageRefPtr& stage, const pxr::SdfPath& path) {
    auto cam = pxr::UsdGeomCamera::Define(stage, path);
    cam.GetFocalLengthAttr().Set(50.0f);
    cam.GetHorizontalApertureAttr().Set(36.0f);
    cam.GetVerticalApertureAttr().Set(24.0f);
    cam.GetClippingRangeAttr().Set(pxr::GfVec2f(0.1f, 10000.0f));
    return cam.GetPrim();
}

std::vector<PrimFactory> CameraAdapter::get_factories() const {
    return {
        {"Cameras", "Camera", "Camera", define_camera},
    };
}

}  // namespace pts::rendering
