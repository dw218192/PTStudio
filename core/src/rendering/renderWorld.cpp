#include <core/rendering/renderWorld.h>

namespace pts::rendering {

void RenderWorld::clear() {
    meshes.clear();
    objects.clear();
    materials.clear();
}

}  // namespace pts::rendering
