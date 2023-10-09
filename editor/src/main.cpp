#include "include/application.h"
#include "include/editorApplication.h"
#include "include/editorRenderer.h"

int main() {
    RenderConfig const config{
        1280, 720,
        60.0, 120.0
    };
    EditorRenderer renderer{ config };

    EditorApplication app{ renderer, "PT Editor" };
    app.run();
}
