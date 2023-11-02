#include "include/editorApplication.h"
#include "include/editorRenderer.h"

int main() {
    auto config = RenderConfig {
        1280, 720,
        60.0, 120.0
    };

    EditorRenderer::create(config);
	EditorApplication::create(EditorRenderer::get(), "PT Editor");
    EditorApplication::get().run();
}
