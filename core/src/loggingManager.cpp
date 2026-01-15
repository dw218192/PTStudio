#include <core/logging.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <mutex>
#include <vector>

namespace pts {

namespace {
void apply_cfg_to_sink(const spdlog::sink_ptr& s, const Config& cfg) {
    s->set_pattern(cfg.pattern);
    // keep sink permissive and let logger level filter
    s->set_level(spdlog::level::trace);
}
}  // namespace

LoggingManager::LoggingManager(Config const& config) : cfg_(config) {
    // create default console sink
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    apply_cfg_to_sink(console, cfg_);
    sinks_.push_back(console);
}

LoggingManager::~LoggingManager() {
    // flush all loggers before shutdown
    spdlog::apply_all([](const std::shared_ptr<spdlog::logger>& l) { l->flush(); });
    // drop all loggers
    spdlog::drop_all();
    spdlog::shutdown();
}

auto LoggingManager::get_logger(std::string_view name) noexcept -> spdlog::logger& {
    return *get_logger_shared(name);
}

auto LoggingManager::get_logger_shared(std::string_view name) noexcept
    -> std::shared_ptr<spdlog::logger> {
    std::lock_guard lk(mtx_);

    if (auto existing = spdlog::get(std::string{name})) {
        return existing;
    }

    auto logger = std::make_shared<spdlog::logger>(std::string{name}, sinks_.begin(), sinks_.end());
    logger->set_level(static_cast<spdlog::level::level_enum>(cfg_.level));
    logger->set_pattern(cfg_.pattern);

    spdlog::register_logger(logger);
    return logger;
}

void LoggingManager::add_sink(spdlog::sink_ptr sink) {
    std::lock_guard lk(mtx_);

    apply_cfg_to_sink(sink, cfg_);

    // add to our sink list for future loggers
    sinks_.push_back(sink);

    // add to existing loggers
    spdlog::apply_all(
        [&](const std::shared_ptr<spdlog::logger>& l) { l->sinks().push_back(sink); });
}

}  // namespace pts
