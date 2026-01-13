#include "core/plugin/pluginManager.h"
#include <spdlog/spdlog.h>
#include <boost/dll/import.hpp>
#include <boost/dll/shared_library.hpp>
#include <boost/function.hpp>
#include <stdexcept>

namespace pts {

// Internal representation of a loaded plugin
struct PluginManager::LoadedPlugin {
    boost::dll::shared_library library;
    const PtsPluginDescriptor* descriptor;
    void* instance;

    LoadedPlugin(boost::dll::shared_library lib, const PtsPluginDescriptor* desc)
        : library(std::move(lib)), descriptor(desc), instance(nullptr) {}
};

PluginManager::PluginManager() {
    spdlog::info("PluginManager initialized");
}

PluginManager::~PluginManager() {
    shutdown();
}

void PluginManager::scan_directory(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        spdlog::warn("Plugin directory does not exist: {}", dir.string());
        return;
    }

    spdlog::info("Scanning for plugins in: {}", dir.string());
    
    int found_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
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
            auto it = std::find_if(plugins_.begin(), plugins_.end(),
                [&](const PluginInfo& p) { return p.id == info.id; });
            
            if (it != plugins_.end()) {
                spdlog::warn("Duplicate plugin ID '{}' found at: {}", info.id, path.string());
                continue;
            }

            plugins_.push_back(std::move(info));
            found_count++;
            spdlog::info("  Found plugin: {} ({}) v{}", 
                info.display_name, info.id, info.version);
        }
    }

    spdlog::info("Scan complete. Found {} plugin(s)", found_count);
}

bool PluginManager::try_load_descriptor(const std::filesystem::path& dll_path, PluginInfo& out_info) {
    try {
        // Temporarily load the library to read its descriptor
        // Boost.DLL requires string path
        boost::dll::shared_library lib(dll_path.string(), boost::dll::load_mode::default_mode);
        
        if (!lib.has("pts_plugin_get_desc")) {
            spdlog::debug("  {} does not export pts_plugin_get_desc", dll_path.filename().string());
            return false;
        }

        // Import the descriptor function
        auto get_desc = lib.get<const PtsPluginDescriptor*()>("pts_plugin_get_desc");
        const PtsPluginDescriptor* desc = get_desc();

        if (!desc) {
            spdlog::error("  {} returned null descriptor", dll_path.filename().string());
            return false;
        }

        // Validate API version
        if (desc->api_version != PTS_PLUGIN_API_VERSION) {
            spdlog::error("  {} has incompatible API version {} (expected {})",
                dll_path.filename().string(), desc->api_version, PTS_PLUGIN_API_VERSION);
            return false;
        }

        // Fill out plugin info
        out_info.id = desc->plugin_id ? desc->plugin_id : "";
        out_info.display_name = desc->display_name ? desc->display_name : "";
        out_info.version = desc->version ? desc->version : "";
        out_info.kind = desc->kind;
        out_info.dll_path = dll_path;
        out_info.is_loaded = false;
        out_info.instance = nullptr;

        if (out_info.id.empty()) {
            spdlog::error("  {} has empty plugin_id", dll_path.filename().string());
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::debug("  Failed to load {}: {}", dll_path.filename().string(), e.what());
        return false;
    }
}

bool PluginManager::load_plugin(const std::string& plugin_id) {
    // Check if already loaded
    if (loaded_plugins_.find(plugin_id) != loaded_plugins_.end()) {
        spdlog::warn("Plugin '{}' is already loaded", plugin_id);
        return false;
    }

    // Find plugin info
    auto it = std::find_if(plugins_.begin(), plugins_.end(),
        [&](const PluginInfo& p) { return p.id == plugin_id; });
    
    if (it == plugins_.end()) {
        spdlog::error("Plugin '{}' not found in registry", plugin_id);
        return false;
    }

    try {
        spdlog::info("Loading plugin: {} ({})", it->display_name, plugin_id);
        
        // Load the library - Boost.DLL requires string path
        boost::dll::shared_library lib(it->dll_path.string(), boost::dll::load_mode::default_mode);
        
        // Get descriptor
        auto get_desc = lib.get<const PtsPluginDescriptor*()>("pts_plugin_get_desc");
        const PtsPluginDescriptor* desc = get_desc();

        // Create loaded plugin entry
        auto loaded = std::make_unique<LoadedPlugin>(std::move(lib), desc);

        // Create instance
        if (desc->create) {
            loaded->instance = desc->create();
            spdlog::debug("  Plugin instance created: {}", static_cast<void*>(loaded->instance));
        }

        // Call on_load
        if (desc->on_load && loaded->instance) {
            desc->on_load(loaded->instance);
            spdlog::debug("  Plugin on_load() called");
        }

        // Update registry
        it->is_loaded = true;
        it->instance = loaded->instance;

        loaded_plugins_[plugin_id] = std::move(loaded);
        
        spdlog::info("Plugin '{}' loaded successfully", plugin_id);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load plugin '{}': {}", plugin_id, e.what());
        return false;
    }
}

void PluginManager::unload_plugin(const std::string& plugin_id) {
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) {
        spdlog::warn("Plugin '{}' is not loaded", plugin_id);
        return;
    }

    spdlog::info("Unloading plugin: {}", plugin_id);

    auto& loaded = it->second;
    const auto* desc = loaded->descriptor;

    // Call on_unload
    if (desc->on_unload && loaded->instance) {
        desc->on_unload(loaded->instance);
        spdlog::debug("  Plugin on_unload() called");
    }

    // Destroy instance
    if (desc->destroy && loaded->instance) {
        desc->destroy(loaded->instance);
        spdlog::debug("  Plugin instance destroyed");
    }

    // Update registry
    auto reg_it = std::find_if(plugins_.begin(), plugins_.end(),
        [&](const PluginInfo& p) { return p.id == plugin_id; });
    if (reg_it != plugins_.end()) {
        reg_it->is_loaded = false;
        reg_it->instance = nullptr;
    }

    // Remove from loaded map (this will unload the DLL)
    loaded_plugins_.erase(it);
    
    spdlog::info("Plugin '{}' unloaded", plugin_id);
}

void* PluginManager::get_plugin_instance(const std::string& plugin_id) const {
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) {
        return nullptr;
    }
    return it->second->instance;
}

void PluginManager::shutdown() {
    spdlog::info("Shutting down PluginManager");
    
    // Unload all plugins in reverse order
    std::vector<std::string> plugin_ids;
    plugin_ids.reserve(loaded_plugins_.size());
    for (const auto& [id, _] : loaded_plugins_) {
        plugin_ids.push_back(id);
    }

    for (auto it = plugin_ids.rbegin(); it != plugin_ids.rend(); ++it) {
        unload_plugin(*it);
    }

    plugins_.clear();
    spdlog::info("PluginManager shutdown complete");
}

} // namespace pts

