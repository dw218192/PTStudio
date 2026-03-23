#pragma once

#include <core/rendering/sceneAdapter.h>

#include <glm/glm.hpp>

namespace pts::rendering {

class CameraAdapter final : public ISceneAdapter {
   public:
    static CameraAdapter& instance();

    bool can_adapt(const pxr::UsdPrim& prim) const override;
    void sync(pxr::UsdPrim prim, SyncScope& scope) override;
    std::vector<PropertyDescriptor> get_properties(const pxr::UsdPrim& prim) const override;
    std::vector<PrimFactory> get_factories() const override;

    /// Create a UsdGeomCamera prim from a view matrix and lens parameters.
    static pxr::UsdPrim create_from_view(const pxr::UsdStageRefPtr& stage, const pxr::SdfPath& path,
                                         const glm::mat4& view_matrix, float fov_y_radians,
                                         float near_clip, float far_clip);

   private:
    CameraAdapter() = default;
};

}  // namespace pts::rendering
