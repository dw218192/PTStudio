#include <core/commandLine.h>
#include <core/diagnostics.h>
#include <core/enumUtils.h>
#include <core/error.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <core/renderConfig.h>

#include <iostream>

#include "editorApplication.h"

int main(int argc, char* argv[]) {
    try {
        // Pre-parse for infrastructure args needed before app construction.
        // Unrecognized args (like --num-frames) are silently ignored.
        pts::CommandLine pre_cli;
        pre_cli.add_string("log-level", "Log level (trace, debug, info, warn, error, critical)");
        pre_cli.add_string("plugins-dir", "Search directory for plugins (relative to executable)");
        pre_cli.parse(argc, argv);

        auto log_level_str = pre_cli.get_string("log-level", "info");
        auto plugins_dir_str = pre_cli.get_string("plugins-dir", "plugins");

        auto opt_log_level = pts::from_string<pts::LogLevel>(log_level_str);
        if (!opt_log_level) {
            std::cerr << "Invalid log level: " << log_level_str << std::endl;
            return static_cast<int>(pts::ErrorCode::InvalidArgument);
        }

        auto render_config = pts::RenderConfig{1280, 720, 60.0, 120.0};

        // Initialize logging manager
        pts::Config logging_config{};
        logging_config.level = *opt_log_level;
        pts::LoggingManager logging_manager{logging_config};

        // Initialize plugin manager
        auto core_logger = logging_manager.get_logger("Core");
        auto plugin_manager_logger = logging_manager.get_logger_shared("PluginManager");
        pts::PluginManager plugin_manager{plugin_manager_logger, logging_manager};

        // Scan and load plugins
        plugin_manager.scan_directory(plugins_dir_str);

        // Auto-load all discovered plugins
        for (const auto& plugin : plugin_manager.get_plugins()) {
            core_logger.info("Discovered plugin: {} ({})", plugin.display_name, plugin.id);
            plugin_manager.load_plugin(plugin.id);
        }

        // Create application, init (register + parse + process args), and run
        pts::editor::EditorApplication app{"Editor", render_config, logging_manager,
                                           plugin_manager};
        if (!app.init(argc, argv)) {
            return 0;  // --help was shown
        }
        app.run();

        // Plugin manager and logging manager will be destroyed here, ensuring proper shutdown
    } catch (std::exception& e) {
        std::cerr << "Exception thrown: " << e.what() << std::endl;
        pts::diagnostics::print_stacktrace();
        return static_cast<int>(pts::ErrorCode::InternalError);
    } catch (...) {
        std::cerr << "Unknown exception thrown" << std::endl;
        pts::diagnostics::print_stacktrace();
        return static_cast<int>(pts::ErrorCode::InternalError);
    }

    return static_cast<int>(pts::ErrorCode::Ok);
}
