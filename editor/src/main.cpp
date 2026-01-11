#include <core/logging.h>
#include <spdlog/spdlog.h>

#include "editorApplication.h"
#include "editorRenderer.h"

int main() {
    auto config = PTS::RenderConfig{1280, 720, 60.0, 120.0};
    auto logging_config = PTS::Logging::Config{};
    logging_config.level = PTS::Logging::LogLevel::Info;
    try {
        PTS::Logging::init_logging(logging_config);
        PTS::Editor::EditorApplication::create("PT Editor", config);
        PTS::Editor::EditorApplication::get().run();
    } catch (std::exception& e) {
        spdlog::error("Exception thrown: {}", e.what());
        return 1;
    }
    return 0;
}
