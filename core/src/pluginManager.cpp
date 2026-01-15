#include <core/logging.h>
#include <core/pluginManager.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <boost/dll/import.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/function.hpp>
#include <core/pluginUtils.hpp>
#include <stdexcept>

namespace pts {

// Global plugin manager instance pointer for host API callbacks
static PluginManager* g_plugin_manager = nullptr;

PluginManager::PluginManager(std::shared_ptr<spdlog::logger> logger,
                             LoggingManager& logging_manager)
    : m_logger(std::move(logger)), m_logging_manager(&logging_manager) {
    setup_host_api();
    g_plugin_manager = this;
    m_logger->info("PluginManager initialized");
}

PluginManager::~PluginManager() {
    g_plugin_manager = nullptr;
    shutdown();
}

void PluginManager::setup_host_api() {
    m_host_api.create_logger = &PluginManager::create_logger_impl;
    m_host_api.log = &PluginManager::log_impl;
    m_host_api.is_level_enabled = &PluginManager::is_level_enabled_impl;
    m_host_api.get_plugin_handle = &PluginManager::get_plugin_handle_impl;
    m_host_api.query_interface = &PluginManager::query_interface_impl;
    m_host_api.render_graph_api = nullptr;
    m_host_api.render_world_api = nullptr;
}

LoggerHandle PluginManager::create_logger_impl(const char* name) {
    if (!g_plugin_manager || !name) {
        return nullptr;
    }

    // Use LoggingManager to get consistent formatting
    return &g_plugin_manager->m_logging_manager->get_logger(name);
}

void PluginManager::log_impl(LoggerHandle logger, PtsLogLevel level, const char* message) {
    if (!logger || !message) {
        return;
    }

    auto* spd_logger = static_cast<spdlog::logger*>(logger);
    spdlog::level::level_enum spd_level = spdlog::level::off;

    switch (level) {
        case PTS_LOG_LEVEL_TRACE:
            spd_level = spdlog::level::trace;
            break;
        case PTS_LOG_LEVEL_DEBUG:
            spd_level = spdlog::level::debug;
            break;
        case PTS_LOG_LEVEL_INFO:
            spd_level = spdlog::level::info;
            break;
        case PTS_LOG_LEVEL_WARNING:
            spd_level = spdlog::level::warn;
            break;
        case PTS_LOG_LEVEL_ERROR:
            spd_level = spdlog::level::err;
            break;
        case PTS_LOG_LEVEL_CRITICAL:
            spd_level = spdlog::level::critical;
            break;
        case PTS_LOG_LEVEL_OFF:
            spd_level = spdlog::level::off;
            break;
        default:
            return;
    }

    spd_logger->log(spd_level, message);
}

bool PluginManager::is_level_enabled_impl(LoggerHandle logger, PtsLogLevel level) {
    if (!logger) {
        return false;
    }

    auto* spd_logger = static_cast<spdlog::logger*>(logger);
    spdlog::level::level_enum spd_level = spdlog::level::off;

    switch (level) {
        case PTS_LOG_LEVEL_TRACE:
            spd_level = spdlog::level::trace;
            break;
        case PTS_LOG_LEVEL_DEBUG:
            spd_level = spdlog::level::debug;
            break;
        case PTS_LOG_LEVEL_INFO:
            spd_level = spdlog::level::info;
            break;
        case PTS_LOG_LEVEL_WARNING:
            spd_level = spdlog::level::warn;
            break;
        case PTS_LOG_LEVEL_ERROR:
            spd_level = spdlog::level::err;
            break;
        case PTS_LOG_LEVEL_CRITICAL:
            spd_level = spdlog::level::critical;
            break;
        case PTS_LOG_LEVEL_OFF:
            spd_level = spdlog::level::off;
            break;
        default:
            return false;
    }

    return spd_logger->should_log(spd_level);
}

PluginHandle PluginManager::get_plugin_handle_impl(const char* plugin_id) {
    if (!g_plugin_manager || !plugin_id) {
        return nullptr;
    }
    return g_plugin_manager->get_plugin_instance(plugin_id);
}

void* PluginManager::query_interface_impl(PluginHandle plugin_handle, const char* iid) {
    if (!g_plugin_manager || !plugin_handle || !iid) {
        return nullptr;
    }

    // Find the loaded plugin descriptor
    for (const auto& loaded : g_plugin_manager->m_loaded_plugins) {
        if (loaded.instance == plugin_handle) {
            if (loaded.descriptor && loaded.descriptor->query_interface) {
                return loaded.descriptor->query_interface(plugin_handle, iid);
            }
            break;
        }
    }
    return nullptr;
}

size_t PluginManager::scan_directory(std::string_view exe_relative_dir) {
    auto exe_dir = boost::dll::program_location().parent_path();
    auto plugin_dir = std::filesystem::path{exe_dir.string()} / exe_relative_dir;

    m_logger->info("Scanning for plugins in: {}", plugin_dir.string());

    if (!std::filesystem::exists(plugin_dir)) {
        m_logger->warn("Plugin directory does not exist: {}", plugin_dir.string());
        return 0;
    }

    size_t found_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator{plugin_dir}) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto& path = entry.path();
        auto ext = path.extension().string();

// Check for DLL/SO/DYLIB extensions
#ifdef _WIN32
        if (ext != ".dll") continue;
#elif __APPLE__
        if (ext != ".dylib") continue;
#else
        if (ext != ".so") continue;
#endif

        PluginInfo info;
        if (try_load_descriptor(path, info)) {
            // Check if we already have this plugin
            auto it = std::find_if(m_plugins.begin(), m_plugins.end(),
                                   [&](const PluginInfo& p) { return p.id == info.id; });

            if (it != m_plugins.end()) {
                m_logger->warn("Duplicate plugin ID '{}' found at: {}", info.id, path.string());
                continue;
            }

            found_count++;
            m_logger->info("  Found plugin: {} ({}) v{}", info.display_name, info.id, info.version);
            m_plugins.push_back(std::move(info));
        }
    }

    m_logger->info("Scan complete. Found {} plugin(s)", found_count);
    return found_count;
}

bool PluginManager::try_load_descriptor(const std::filesystem::path& dll_path,
                                        PluginInfo& out_info) {
    try {
        // Temporarily load the library to read its descriptor
        // Boost.DLL requires string path
        boost::dll::shared_library lib(dll_path.string(), boost::dll::load_mode::default_mode);

        if (!lib.has(PLUGIN_ENTRY_POINT_NAME)) {
            m_logger->debug("  {} does not export " PLUGIN_ENTRY_POINT_NAME,
                            dll_path.filename().string());
            return false;
        }

        // Import the descriptor function
        auto get_desc = lib.get<const PtsPluginDescriptor*()>(PLUGIN_ENTRY_POINT_NAME);
        const PtsPluginDescriptor* desc = get_desc();

        if (!desc) {
            m_logger->error("  {} returned null descriptor", dll_path.filename().string());
            return false;
        }

        // Validate API version
        if (desc->api_version != PTS_PLUGIN_API_VERSION) {
            m_logger->error("  {} has incompatible API version {} (expected {})",
                            dll_path.filename().string(), desc->api_version,
                            PTS_PLUGIN_API_VERSION);
            return false;
        }

        // Validate struct size
        if (desc->struct_size != sizeof(PtsPluginDescriptor)) {
            m_logger->error("  {} has incompatible ABI (struct size mismatch) {} (expected {})",
                            dll_path.filename().string(), desc->struct_size,
                            sizeof(PtsPluginDescriptor));
            return false;
        }

        // Fill out plugin info
        out_info.id = desc->plugin_id ? desc->plugin_id : "<unknown>";
        out_info.display_name = desc->display_name ? desc->display_name : "<unknown>";
        out_info.version = desc->version ? desc->version : "<unknown>";
        out_info.kind = desc->kind;
        out_info.dll_path = dll_path;
        out_info.is_loaded = false;
        out_info.instance = nullptr;

        if (out_info.id.empty()) {
            m_logger->error("  {} has empty plugin_id", dll_path.filename().string());
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        m_logger->debug("  Failed to load {}: {}", dll_path.filename().string(), e.what());
        return false;
    }
}

bool PluginManager::load_plugin(std::string_view plugin_id) {
    if (find_loaded_plugin(plugin_id) != m_loaded_plugins.end()) {
        m_logger->warn("Plugin '{}' is already loaded", plugin_id);
        return false;
    }

    // Find plugin info
    auto it = std::find_if(m_plugins.begin(), m_plugins.end(),
                           [&](const PluginInfo& p) { return p.id == plugin_id; });

    if (it == m_plugins.end()) {
        m_logger->error("Plugin '{}' not found in registry", plugin_id);
        return false;
    }

    try {
        m_logger->info("Loading plugin: {} ({})", it->display_name, plugin_id);

        // Load the library - Boost.DLL requires string path
        boost::dll::shared_library lib(it->dll_path.string(), boost::dll::load_mode::default_mode);

        // Get descriptor
        auto get_desc = lib.get<const PtsPluginDescriptor*()>(PLUGIN_ENTRY_POINT_NAME);
        const PtsPluginDescriptor* desc = get_desc();

        // Create loaded plugin entry
        auto& loaded = m_loaded_plugins.emplace_back(std::string{plugin_id}, std::move(lib), desc);

        // Create instance
        if (desc->create) {
            loaded.instance = desc->create(&m_host_api);
            m_logger->debug("  Plugin instance created: {}", static_cast<void*>(loaded.instance));
        }

        // Call on_load - this is a potentially fallible operation
        if (desc->on_load && loaded.instance) {
            bool load_success = desc->on_load(loaded.instance);
            m_logger->debug("  Plugin on_load() called, result: {}", load_success);

            if (!load_success) {
                m_logger->error("Plugin '{}' on_load() failed, cleaning up", plugin_id);

                // Cleanup: destroy instance
                if (desc->destroy) {
                    desc->destroy(loaded.instance);
                    m_logger->debug("  Plugin instance destroyed after on_load failure");
                }

                // Remove from loaded plugins vector
                m_loaded_plugins.pop_back();

                return false;
            }
        }

        // Update registry
        it->is_loaded = true;
        it->instance = loaded.instance;

        m_logger->info("Plugin '{}' loaded successfully", plugin_id);
        return true;

    } catch (const std::exception& e) {
        m_logger->error("Failed to load plugin '{}': {}", plugin_id, e.what());
        return false;
    }
}

void PluginManager::unload_plugin(std::string_view plugin_id) {
    auto loaded = find_loaded_plugin(plugin_id);
    if (loaded == m_loaded_plugins.end()) {
        m_logger->warn("Plugin '{}' is not loaded", plugin_id);
        return;
    }

    m_logger->info("Unloading plugin: {}", plugin_id);

    const auto desc = loaded->descriptor;

    // Call on_unload
    if (desc->on_unload && loaded->instance) {
        desc->on_unload(loaded->instance);
        m_logger->debug("  Plugin on_unload() called");
    }

    // Destroy instance
    if (desc->destroy && loaded->instance) {
        desc->destroy(loaded->instance);
        m_logger->debug("  Plugin instance destroyed");
    }

    // Update registry
    auto reg_it = std::find_if(m_plugins.begin(), m_plugins.end(),
                               [&](const PluginInfo& p) { return p.id == plugin_id; });
    if (reg_it != m_plugins.end()) {
        reg_it->is_loaded = false;
        reg_it->instance = nullptr;
    }

    // Remove from loaded vector
    m_loaded_plugins.erase(loaded);
    m_logger->info("Plugin '{}' unloaded", plugin_id);
}

void* PluginManager::get_plugin_instance(std::string_view plugin_id) const {
    auto loaded = find_loaded_plugin(plugin_id);
    return loaded != m_loaded_plugins.end() ? loaded->instance : nullptr;
}

void* PluginManager::query_interface(void* plugin_handle, const char* interface_id) const {
    if (!plugin_handle || !interface_id) {
        return nullptr;
    }

    // Find the loaded plugin with this instance
    for (const auto& loaded : m_loaded_plugins) {
        if (loaded.instance == plugin_handle) {
            if (loaded.descriptor && loaded.descriptor->query_interface) {
                return loaded.descriptor->query_interface(plugin_handle, interface_id);
            }
            break;
        }
    }
    return nullptr;
}

void PluginManager::shutdown() {
    m_logger->info("Shutting down PluginManager");

    // Unload all plugins in reverse order
    std::vector<std::string> plugin_ids;
    plugin_ids.reserve(m_loaded_plugins.size());
    for (const auto& loaded : m_loaded_plugins) {
        plugin_ids.push_back(loaded.plugin_id);
    }

    for (auto it = plugin_ids.rbegin(); it != plugin_ids.rend(); ++it) {
        unload_plugin(*it);
    }

    m_plugins.clear();
    m_logger->info("PluginManager shutdown complete");
}
auto PluginManager::find_loaded_plugin(std::string_view plugin_id)
    -> std::vector<LoadedPlugin>::iterator {
    return std::find_if(m_loaded_plugins.begin(), m_loaded_plugins.end(),
                        [&](const LoadedPlugin& p) { return p.plugin_id == plugin_id; });
}

}  // namespace pts
