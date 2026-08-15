#include <core/commandLine.h>
#include <core/diagnostics.h>
#include <core/enumUtils.h>
#include <core/error.h>
#include <core/loggingManager.h>

#include <iostream>

#include "editorApplication.h"

int main(int argc, char* argv[]) {
    try {
        // Pre-parse for log-level (needed before LoggingManager construction).
        // Don't handle --help here -- let the full CLI in app.init() handle it
        // so all registered options are visible.
        pts::CommandLine pre_cli;
        pre_cli.add_string("log-level", "Log level (trace, debug, info, warn, error, critical)");
        UNUSED(pre_cli.parse(argc, argv));  // -h handled by app.init()

        auto log_level_str = pre_cli.get_string("log-level", "info");

        auto opt_log_level = pts::from_string<pts::LogLevel>(log_level_str);
        if (!opt_log_level) {
            std::cerr << "Invalid log level: " << log_level_str << std::endl;
            return static_cast<int>(pts::ErrorCode::InvalidArgument);
        }

        // Initialize logging manager
        pts::Config logging_config{};
        logging_config.level = *opt_log_level;
        pts::LoggingManager logging_manager{logging_config};

        // Create application, init (register + parse + process args), and run
        pts::editor::EditorApplication app{"Editor", logging_manager};
        if (auto exit_code = app.init(argc, argv)) {
            return *exit_code;
        }
        app.run();
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
