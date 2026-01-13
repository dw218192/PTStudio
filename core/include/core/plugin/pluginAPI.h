#pragma once

/**
 * PTS Plugin API v1
 * 
 * C ABI for stable plugin interface across compilers and build configurations.
 * All functions use C linkage and POD types only.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// ============================================================================
// Version and Identification
// ============================================================================

#define PTS_PLUGIN_API_VERSION 1

typedef enum {
    PTS_PLUGIN_KIND_UNKNOWN = 0,
    PTS_PLUGIN_KIND_RENDERER = 1,
    PTS_PLUGIN_KIND_TEST = 99,  // For testing/demo purposes
} PtsPluginKind;

// ============================================================================
// Plugin Descriptor (returned by pts_plugin_get_desc)
// ============================================================================

typedef struct {
    uint32_t api_version;           // Must be PTS_PLUGIN_API_VERSION
    PtsPluginKind kind;             // Plugin category
    const char* plugin_id;          // Unique identifier (e.g., "test_plugin")
    const char* display_name;       // Human-readable name
    const char* version;            // Plugin version string
    
    // Lifecycle callbacks (opaque handle returned)
    void* (*create)(void);          // Create plugin instance
    void (*destroy)(void* instance); // Destroy plugin instance
    void (*on_load)(void* instance); // Called after creation
    void (*on_unload)(void* instance); // Called before destruction
} PtsPluginDescriptor;

// ============================================================================
// Entry Point (required export)
// ============================================================================

/**
 * Every plugin DLL must export this function.
 * Returns a pointer to a static descriptor (valid for plugin lifetime).
 */
#ifdef _WIN32
    #define PTS_PLUGIN_EXPORT __declspec(dllexport)
#else
    #define PTS_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

// The plugin must implement this:
// PTS_PLUGIN_EXPORT const PtsPluginDescriptor* pts_plugin_get_desc(void);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ Helper Macros (optional, for plugin implementers)
// ============================================================================

#ifdef __cplusplus

/**
 * Convenience macro to implement the plugin entry point.
 * 
 * Usage in plugin .cpp:
 *   PTS_PLUGIN_DEFINE(MyPluginClass, PTS_PLUGIN_KIND_TEST, "my_plugin", "My Plugin", "1.0.0")
 */
#define PTS_PLUGIN_DEFINE(PluginClass, Kind, Id, Name, Version) \
    extern "C" { \
        PTS_PLUGIN_EXPORT const PtsPluginDescriptor* pts_plugin_get_desc(void) { \
            static PtsPluginDescriptor desc = { \
                PTS_PLUGIN_API_VERSION, \
                Kind, \
                Id, \
                Name, \
                Version, \
                [](void) -> void* { return new PluginClass(); }, \
                [](void* p) { delete static_cast<PluginClass*>(p); }, \
                [](void* p) { static_cast<PluginClass*>(p)->on_load(); }, \
                [](void* p) { static_cast<PluginClass*>(p)->on_unload(); } \
            }; \
            return &desc; \
        } \
    }

/**
 * Base interface for C++ plugin implementations.
 * Plugins can inherit from this for convenience.
 */
struct IPlugin {
    virtual ~IPlugin() = default;
    virtual void on_load() = 0;
    virtual void on_unload() = 0;
};

#endif // __cplusplus
