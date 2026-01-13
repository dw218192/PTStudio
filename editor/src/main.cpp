#include <core/logging.h>
#include <core/plugin/pluginManager.h>
#include <spdlog/spdlog.h>
#include <filesystem>

#include "editorApplication.h"
#include "editorRenderer.h"

int main() {
    auto config = PTS::RenderConfig{1280, 720, 60.0, 120.0};
    auto logging_config = PTS::Logging::Config{};
    logging_config.level = PTS::Logging::LogLevel::Info;
    try {
        PTS::Logging::init_logging(logging_config);
        
        // Initialize plugin system
        pts::PluginManager plugin_manager;
        
        // Get plugins directory (relative to executable)
        std::filesystem::path plugins_dir = std::filesystem::current_path() / "plugins";
        
        // Scan and load plugins
        if (std::filesystem::exists(plugins_dir)) {
            plugin_manager.scan_directory(plugins_dir);
            
            // Auto-load all discovered plugins
            for (const auto& plugin : plugin_manager.get_plugins()) {
                spdlog::info("Discovered plugin: {} ({})", plugin.display_name, plugin.id);
                plugin_manager.load_plugin(plugin.id);
            }
        } else {
            spdlog::warn("Plugins directory not found: {}", plugins_dir.string());
            spdlog::info("Run 'pts package' to deploy plugins");
        }
        
        // Run the application
        PTS::Editor::EditorApplication::create("PT Editor", config);
        PTS::Editor::EditorApplication::get().run();
        
        // Cleanup plugins on exit
        plugin_manager.shutdown();
        
    } catch (std::exception& e) {
        spdlog::error("Exception thrown: {}", e.what());
        return 1;
    }
    return 0;
}
