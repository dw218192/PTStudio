#include "include/application.h"
#include "include/editorApplication.h"
#include "include/editorRenderer.h"

int main() {
    auto config = RenderConfig {
        1280, 720,
        60.0, 120.0
    };
    auto renderer = EditorRenderer{ config };
    auto scene = Application::check_error(
        Scene::from_obj_file("D:/Repos/PTStudio/_files/ada.obj")
    );
    auto app = EditorApplication { renderer, scene, "PT Editor" };
    app.run();
}
