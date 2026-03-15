#include <core/rendering/renderWorld.h>

namespace pts::rendering {

void RenderWorld::clear() {
    meshes.clear();
    objects.clear();
    materials.clear();
    lights.clear();
    material_cache.clear();
}

}  // namespace pts::rendering
