#include <core/guiApplication.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <imgui.h>

class HelloApp : public pts::GUIApplication {
   public:
    HelloApp(pts::LoggingManager& logging_manager, pts::PluginManager& plugin_manager)
        : pts::GUIApplication("Hello", logging_manager, plugin_manager, 1280, 720, 1.0f / 60.0f) {
    }

    void loop(float dt) override {
        static_cast<void>(dt);
        ImGui::ShowDemoWindow();
    }
};

int main(int argc, char* argv[]) {
    pts::Config config;
    config.level = pts::LogLevel::Info;
    config.pattern = "[%H:%M:%S] [%^%L%$] [%n] %v";

    pts::LoggingManager logging_manager(config);
    auto logger = logging_manager.get_logger_shared("Hello");
    pts::PluginManager plugin_manager(logger, logging_manager);

    try {
        HelloApp app(logging_manager, plugin_manager);
        if (!app.init(argc, argv)) {
            return 0;
        }
        app.run();
    } catch (const std::exception& e) {
        logging_manager.get_logger().error("Application error: {}", e.what());
        return 1;
    }

    return 0;
}
