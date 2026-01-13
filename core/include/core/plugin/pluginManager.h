#pragma once

#include "pluginAPI.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace pts {

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
};

/**
 * Manages plugin discovery, loading, and lifecycle.
 * Uses Boost.DLL for cross-platform dynamic loading.
 */
class PluginManager {
public:
    PluginManager();
    ~PluginManager();

    // Disable copying
    PluginManager(const PluginManager&) = delete;
    PluginManager& operator=(const PluginManager&) = delete;

    /**
     * Scan a directory for plugin DLLs (*.dll/*.so/*.dylib).
     * Does not load them, just discovers available plugins.
     */
    void scan_directory(const std::filesystem::path& dir);

    /**
     * Get list of all discovered plugins (loaded or not).
     */
    const std::vector<PluginInfo>& get_plugins() const { return plugins_; }

    /**
     * Load a specific plugin by ID.
     * Returns true if successful, false if already loaded or error.
     */
    bool load_plugin(const std::string& plugin_id);

    /**
     * Unload a specific plugin by ID.
     */
    void unload_plugin(const std::string& plugin_id);

    /**
     * Get plugin instance by ID (returns nullptr if not loaded).
     */
    void* get_plugin_instance(const std::string& plugin_id) const;

    /**
     * Unload all plugins and clear registry.
     */
    void shutdown();

private:
    struct LoadedPlugin;
    
    std::vector<PluginInfo> plugins_;
    std::unordered_map<std::string, std::unique_ptr<LoadedPlugin>> loaded_plugins_;
    
    bool try_load_descriptor(const std::filesystem::path& dll_path, PluginInfo& out_info);
};

} // namespace pts

