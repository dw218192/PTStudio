#include <core/error.h>
#include <core/loggingManager.h>
#include <core/pluginManager.h>
#include <core/renderConfig.h>
#include <core/utils.h>
#include <spdlog/spdlog.h>

#include <boost/program_options.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <filesystem>
#include <iostream>

#include "editorApplication.h"

namespace po = boost::program_options;

pts::editor::AppConfig make_app_config(const boost::program_options::variables_map& vm) {
    pts::editor::AppConfig c;
    if (vm.count("quit-on-start")) c.quit_on_start = vm["quit-on-start"].as<bool>();
    return c;
}

int main(int argc, char* argv[]) {
    try {
        po::options_description desc("PTStudio Editor Options");

        // clang-format off
        desc.add_options()
            ("help,h", "produce help message")
            ("log-level", po::value<std::string>(), "Log level (trace, debug, info, warn, error, critical)")
            ("plugins-dir", po::value<std::string>(), "Search directory for plugins (relative to executable)")
            ("quit-on-start", po::bool_switch(), "Quit the application after starting, useful for testing");
        // clang-format on

        po::variables_map vm;
        auto parsed_options = po::command_line_parser(argc, argv)
                                  .options(desc)
                                  .style(po::command_line_style::default_style &
                                         ~po::command_line_style::allow_guessing)
                                  .allow_unregistered()
                                  .run();

        po::store(parsed_options, vm);
        po::notify(vm);
        auto unknown_args =
            po::collect_unrecognized(parsed_options.options, po::include_positional);
        if (!unknown_args.empty()) {
            std::cerr << "Ignoring unknown arguments:";
            for (const auto& arg : unknown_args) {
                std::cerr << " " << arg;
            }
            std::cerr << std::endl;
        }
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        std::string log_level_str =
            vm.count("log-level") ? vm["log-level"].as<std::string>() : "info";
        std::string plugins_dir_str =
            vm.count("plugins-dir") ? vm["plugins-dir"].as<std::string>() : "plugins";
        auto app_config = make_app_config(vm);
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

        // Run the application
        pts::editor::EditorApplication{"Editor", render_config, app_config, logging_manager,
                                       plugin_manager}
            .run();

        // Plugin manager and logging manager will be destroyed here, ensuring proper shutdown
    } catch (std::exception& e) {
        std::cerr << "Exception thrown: " << e.what() << std::endl;
        std::cerr << boost::stacktrace::stacktrace() << std::endl;
        return static_cast<int>(pts::ErrorCode::InternalError);
    } catch (boost::exception& e) {
        std::cerr << "Boost exception thrown: " << boost::diagnostic_information(e) << std::endl;
        std::cerr << boost::stacktrace::stacktrace() << std::endl;
        return static_cast<int>(pts::ErrorCode::InternalError);
    } catch (...) {
        std::cerr << "Unknown exception thrown" << std::endl;
        std::cerr << boost::stacktrace::stacktrace() << std::endl;
        return static_cast<int>(pts::ErrorCode::InternalError);
    }

    return static_cast<int>(pts::ErrorCode::Ok);
}
