#include "include/editorApplication.h"
#include "include/editorRenderer.h"

int main() {
    auto config = PTS::RenderConfig {
        1280, 720,
        60.0, 120.0
    };
    PTS::Editor::EditorApplication::create("PT Editor", config);
    PTS::Editor::EditorApplication::get().run();
}
