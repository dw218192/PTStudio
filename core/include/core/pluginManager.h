#pragma once

#include <core/defines.h>
#include <core/plugin.h>
#include <spdlog/spdlog.h>

#include <boost/dll/shared_library.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace pts {

// Registers a statically linked plugin descriptor.
void register_static_plugin(const PtsPluginDescriptor* descriptor) noexcept;

// Internal representation of a loaded plugin
struct LoadedPlugin {
    std::string plugin_id;
    std::unique_ptr<boost::dll::shared_library> library;
    const PtsPluginDescriptor* descriptor;
    void* instance;

    LoadedPlugin(std::string plugin_id, boost::dll::shared_library library,
                 const PtsPluginDescriptor* descriptor) noexcept
        : plugin_id(std::move(plugin_id)),
          library(std::make_unique<boost::dll::shared_library>(std::move(library))),
          descriptor(descriptor),
          instance(nullptr) {
    }

    LoadedPlugin(std::string plugin_id, const PtsPluginDescriptor* descriptor) noexcept
        : plugin_id(std::move(plugin_id)),
          library(nullptr),
          descriptor(descriptor),
          instance(nullptr) {
    }
};

/**
 * Information about a loaded plugin.
 */
struct PluginInfo {
    std::string id;
    std::string display_name;
    std::string version;
    PtsPluginKind kind;
    std::filesystem::path dll_path;
    bool is_loaded;
    void* instance;  // Opaque plugin instance handle
    bool is_static;
    const PtsPluginDescriptor* static_descriptor;
};

/**
 * Manages plugin discovery, loading, and lifecycle. Also acts as an API bridge between the plugin
 * and the host application. Uses Boost.DLL for cross-platform dynamic loading.
 */
class LoggingManager;

class PluginManager {
   public:
    PluginManager(std::shared_ptr<spdlog::logger> logger, LoggingManager& logging_manager);
    ~PluginManager();

    // Disable copying
    NO_COPY_MOVE(PluginManager);

    /**
     * @brief Scan a directory for plugin DLLs (*.dll/*.so/*.dylib).
     * Does not load them, just discovers available plugins.
     *
     * @param exe_relative_dir The directory relative to the executable to scan for plugins.
     * @return The number of plugins found.
     */
    size_t scan_directory(std::string_view exe_relative_dir);

    /**
     * Get list of all registered plugins (loaded or not).
     */
    const std::vector<PluginInfo>& get_plugins() const {
        return m_plugins;
    }

    /**
     * Load a specific plugin by ID.
     * Returns true if successful, false if already loaded or error.
     */
    bool load_plugin(std::string_view plugin_id);

    /**
     * Unload a specific plugin by ID.
     */
    void unload_plugin(std::string_view plugin_id);

    /**
     * Get plugin instance by ID (returns nullptr if not loaded).
     */
    void* get_plugin_instance(std::string_view plugin_id) const;

    /**
     * Query an interface from a plugin instance.
     * @param plugin_handle The plugin instance handle
     * @param interface_id The interface identifier string
     * @return Pointer to the interface, or nullptr if not found
     */
    void* query_interface(void* plugin_handle, const char* interface_id) const;

    /**
     * Unload all plugins and clear registry.
     */
    void shutdown();

    /**
     * Register a statically linked plugin descriptor.
     */
    bool register_static_plugin(const PtsPluginDescriptor* descriptor);

   private:
    std::shared_ptr<spdlog::logger> m_logger;
    LoggingManager* m_logging_manager;
    std::vector<PluginInfo> m_plugins;
    std::vector<LoadedPlugin> m_loaded_plugins;
    PtsHostApi m_host_api;

    auto find_loaded_plugin(std::string_view plugin_id) -> std::vector<LoadedPlugin>::iterator;
    auto find_loaded_plugin(std::string_view plugin_id) const
        -> std::vector<LoadedPlugin>::const_iterator {
        return const_cast<PluginManager*>(this)->find_loaded_plugin(plugin_id);
    }
    bool try_load_descriptor(const std::filesystem::path& dll_path, PluginInfo& out_info);
    void register_static_plugins_from_registry();

    // Host API implementation helpers
    void setup_host_api();
    static LoggerHandle create_logger_impl(const char* name);
    static void log_impl(LoggerHandle logger, PtsLogLevel level, const char* message);
    static bool is_level_enabled_impl(LoggerHandle logger, PtsLogLevel level);
    static PluginHandle get_plugin_handle_impl(const char* plugin_id);
    static void* query_interface_impl(PluginHandle plugin_handle, const char* iid);
};

}  // namespace pts
