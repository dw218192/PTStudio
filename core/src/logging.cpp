#include "logging.h"

#include <spdlog/sinks/stdout_color_sinks.h>

#include <mutex>

namespace PTS {
namespace Logging {

namespace {

std::mutex g_mtx;
Config g_cfg{};
std::vector<spdlog::sink_ptr> g_sinks;
bool g_inited = false;

void apply_cfg_to_sink(const spdlog::sink_ptr& s) {
    s->set_pattern(g_cfg.pattern);
    // keep sink permissive and let logger level filter
    s->set_level(spdlog::level::trace);
}

void ensure_default_sink() {
    if (g_sinks.empty()) {
        auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        g_sinks.push_back(console);
    }
}

}  // namespace
void init_logging(Config const& config) {
    std::lock_guard lk(g_mtx);

    g_cfg = config;
    ensure_default_sink();

    for (auto& s : g_sinks) apply_cfg_to_sink(s);

    // update existing loggers too
    spdlog::apply_all([&](const std::shared_ptr<spdlog::logger>& l) {
        l->set_level(static_cast<spdlog::level::level_enum>(g_cfg.level));
        l->set_pattern(g_cfg.pattern);
    });

    g_inited = true;
}

void add_sink(spdlog::sink_ptr sink) {
    if (!g_inited) {
        spdlog::warn("add_sink() called before init_logging(), sink will not be applied");
        return;
    }

    std::lock_guard lk(g_mtx);

    // apply current config (or defaults if init_logging not called yet)
    apply_cfg_to_sink(sink);

    // make future loggers inherit it
    g_sinks.push_back(sink);
    spdlog::apply_all(
        [&](const std::shared_ptr<spdlog::logger>& l) { l->sinks().push_back(sink); });
}

auto get_logger(std::string_view name) noexcept -> spdlog::logger& {
    std::lock_guard lk(g_mtx);

    if (!g_inited) {
        ensure_default_sink();
        for (auto& s : g_sinks) apply_cfg_to_sink(s);
        g_inited = true;
    }

    if (auto existing = spdlog::get(std::string{name})) {
        return *existing;
    }

    auto logger =
        std::make_shared<spdlog::logger>(std::string{name}, g_sinks.begin(), g_sinks.end());
    logger->set_level(static_cast<spdlog::level::level_enum>(g_cfg.level));
    // optional; sinks already have pattern, but keeps behavior consistent
    logger->set_pattern(g_cfg.pattern);

    spdlog::register_logger(logger);
    return *logger;
}
}  // namespace Logging
}  // namespace PTS
