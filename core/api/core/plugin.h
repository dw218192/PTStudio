#pragma once

/**
 * Simple lightweight C++ plugin API.
 *  - No plugin dependency system (plugin-level loading ordering is unspecified)
 *  - Fixed plugin kinds that determine loading order
 *  - Supports plugin interfaces for public API exposure via function tables
 *
 * Interface Query Mechanism:
 *  - Each plugin can expose multiple function table interfaces
 *  - Interfaces are identified by unique string IDs (hashed at compile-time)
 *  - Host queries interfaces via query_interface(handle, interface_id)
 *  - Function tables must be C-compatible POD structs with function pointers
 */

#include <stdint.h>

// ============================================================================
// Version and Identification
// ============================================================================

#define PTS_PLUGIN_API_VERSION 1

typedef void* PluginHandle;
typedef void* LoggerHandle;

typedef enum {
    PTS_LOG_LEVEL_TRACE = 0,
    PTS_LOG_LEVEL_DEBUG = 1,
    PTS_LOG_LEVEL_INFO = 2,
    PTS_LOG_LEVEL_WARNING = 3,
    PTS_LOG_LEVEL_ERROR = 4,
    PTS_LOG_LEVEL_CRITICAL = 5,
    PTS_LOG_LEVEL_OFF = 6,
} PtsLogLevel;

typedef enum {
    PTS_PLUGIN_KIND_SUBSYSTEM = 0,
    PTS_PLUGIN_KIND_RENDERER = 1,
} PtsPluginKind;

/**
 * Host API, providing essential methods to interact with the host application.
 * This interface exposes logging and plugin manager methods in an ABI-stable manner.
 */
typedef struct {
    LoggerHandle (*create_logger)(const char* name);
    void (*log)(LoggerHandle logger, PtsLogLevel level, const char* message);
    bool (*is_level_enabled)(LoggerHandle logger, PtsLogLevel level);

    PluginHandle (*get_plugin_handle)(const char* plugin_id);
    void* (*query_interface)(PluginHandle plugin_handle, const char* iid);

    // rendering APIs
    const void* render_graph_api;  // (PtsRenderGraphApi*)
    const void* render_world_api;  // (PtsRenderWorldApi*)
} PtsHostApi;

/**
 * Descriptor for a plugin.
 */
typedef struct {
    uint32_t api_version;      // Must be PTS_PLUGIN_API_VERSION
    uint32_t struct_size;      // Size of the struct in bytes
    PtsPluginKind kind;        // Plugin category
    const char* plugin_id;     // Unique identifier
    const char* display_name;  // Human-readable name
    const char* version;       // Plugin version string

    // Lifecycle callbacks (opaque handle returned)
    PluginHandle (*create)(PtsHostApi* host_api);             // Create plugin instance
    void (*destroy)(PluginHandle instance);                   // Destroy plugin instance
    bool (*on_load)(PluginHandle instance);                   // Called after creation
    void (*on_unload)(PluginHandle instance);                 // Called before destruction
    void* (*query_interface)(PluginHandle, const char* iid);  // Query interface by ID string
} PtsPluginDescriptor;

// ============================================================================
// Entry Point (required export)
// ============================================================================
// Symbol Visibility
// ============================================================================
// Plugins are configured to hide all symbols by default (CXX_VISIBILITY_PRESET=hidden).
// Only symbols explicitly marked with PTS_PLUGIN_EXPORT are made visible.
// This prevents symbol conflicts between plugins and reduces the dynamic symbol table size.

#ifdef _WIN32
#define PTS_PLUGIN_EXPORT __declspec(dllexport)
#else
#define PTS_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/**
 * Every plugin DLL must export this function.
 * Returns a pointer to a static descriptor (valid for plugin lifetime).
 */
#define PLUGIN_ENTRY_POINT_NAME "pts_plugin_get_desc"
