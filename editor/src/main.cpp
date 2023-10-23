#include "include/editorApplication.h"
#include "include/editorRenderer.h"

int main() {
    auto config = RenderConfig {
        1280, 720,
        60.0, 120.0
    };

    EditorRenderer::create(config);
    auto scene = EditorApplication::check_error(Scene::make_triangle_scene());
	EditorApplication::create(EditorRenderer::get(), scene, "PT Editor");
    EditorApplication::get().run();
}
