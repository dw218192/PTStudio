#include <core/error.h>
#include <core/legacy/renderConfig.h>
#include <core/logging.h>
#include <core/pluginManager.h>
#include <core/utils.h>
#include <spdlog/spdlog.h>

#include <boost/program_options.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <filesystem>

#include "editorApplication.h"
#include "editorRenderer.h"

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    po::options_description desc("PTStudio Editor Options");

    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("log-level", po::value<std::string>(), "Log level (trace, debug, info, warn, error, critical)")
        ("plugins-dir", po::value<std::string>(), "Search directory for plugins (relative to executable)");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::string log_level_str = vm.count("log-level") ? vm["log-level"].as<std::string>() : "info";
    std::string plugins_dir_str =
        vm.count("plugins-dir") ? vm["plugins-dir"].as<std::string>() : "plugins";

    auto opt_log_level = pts::from_string<pts::LogLevel>(log_level_str);
    if (!opt_log_level) {
        std::cerr << "Invalid log level: " << log_level_str << std::endl;
        return static_cast<int>(ErrorCode::InvalidArgument);
    }

    auto config = PTS::RenderConfig{1280, 720, 60.0, 120.0};

    try {
        // Initialize logging manager
        pts::Config logging_config{};
        logging_config.level = *opt_log_level;
        pts::LoggingManager logging_manager{logging_config};

        // Initialize plugin manager
        auto core_logger = logging_manager.get_logger("Core");
        auto plugin_manager_logger = logging_manager.get_logger_shared("PluginManager");
        pts::PluginManager plugin_manager{plugin_manager_logger};

        // Scan and load plugins
        plugin_manager.scan_directory(plugins_dir_str);

        // Auto-load all discovered plugins
        for (const auto& plugin : plugin_manager.get_plugins()) {
            core_logger.info("Discovered plugin: {} ({})", plugin.display_name, plugin.id);
            plugin_manager.load_plugin(plugin.id);
        }

        // Run the application
        PTS::Editor::EditorApplication{"PT Editor", config, logging_manager}.run();

        // Plugin manager and logging manager will be destroyed here, ensuring proper shutdown
    } catch (std::exception& e) {
        std::cerr << "Exception thrown: " << e.what() << std::endl;
        std::cerr << boost::stacktrace::stacktrace() << std::endl;
        return static_cast<int>(ErrorCode::InternalError);
    } catch (...) {
        std::cerr << "Unknown exception thrown" << std::endl;
        std::cerr << boost::stacktrace::stacktrace() << std::endl;
        return static_cast<int>(ErrorCode::InternalError);
    }

    return static_cast<int>(ErrorCode::Ok);
}
