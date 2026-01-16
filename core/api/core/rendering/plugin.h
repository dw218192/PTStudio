#pragma once
#include <stdint.h>

#include "../plugin.h"
#include "../types.h"
#include "types.h"

// ============================================================================
// Renderer Plugin Interface
// ============================================================================
//
// This interface defines the contract between the host and renderer plugins.
// Renderer plugins implement rendering techniques (deferred, forward, ray-traced,
// etc.) using the render graph API provided by the host.
//
// Lifecycle:
//   1. Plugin is loaded and create() is called
//   2. on_load() is called for initialization
//   3. Each frame: build_graph() is called to construct the frame's render graph
//   4. on_unload() is called before destruction
//   5. destroy() is called
//
// Threading:
//   - build_graph() is called on the render thread
//   - Other callbacks may be called from any thread (with synchronization)
//

#define RENDERER_PLUGIN_INTERFACE_V1_ID "renderer.interface.v1"

// ============================================================================
// View Parameters
// ============================================================================
//
// Per-view rendering parameters. Keep POD and ABI-stable.
// Add new fields only at the end in future versions.
//

typedef struct PtsViewParams {
    // Camera matrices (column-major)
    float view[16];
    float proj[16];
    float view_proj[16];
    float prev_view_proj[16];

    // Camera position in world space
    float camera_pos[3];
    float _pad0;

    // TAA jitter offset in pixels
    float jitter_xy[2];

    // Timing
    float dt_seconds;
    uint32_t frame_index;  // For noise/TAA sequences (wraps)

    // Viewport dimensions
    uint32_t viewport_w;
    uint32_t viewport_h;

    // Near/far planes
    float near_plane;
    float far_plane;
} PtsViewParams;

// ============================================================================
// Frame Parameters
// ============================================================================

typedef struct PtsFrameParams {
    uint64_t frame_index;  // Monotonically increasing frame counter
    double time_seconds;   // Time since application start
    double wall_time;      // Wall clock time (for time-of-day, etc.)
} PtsFrameParams;

// ============================================================================
// Frame I/O
// ============================================================================
//
// Input/output textures for the frame. These are host-owned resources that
// the plugin should read from or write to.
//
// All handles are external textures that must be imported into the graph
// before use in passes.
//
// Optional textures have null handle ({0}) when not available.
//

typedef struct PtsFrameIO {
    PtsTexture output;   // [Required] Final output (swapchain or offscreen target)
    PtsTexture history;  // [Optional] Previous frame's output (for TAA)
    PtsTexture depth;    // [Optional] Shared depth buffer
    PtsTexture motion;   // [Optional] Motion vectors (for TAA/motion blur)
    PtsTexture normal;   // [Optional] G-buffer normals
    PtsTexture albedo;   // [Optional] G-buffer albedo
} PtsFrameIO;

// ============================================================================
// Debug Output Entry
// ============================================================================
//
// Used by get_debug_outputs() to expose intermediate buffers for visualization.
//

typedef struct PtsDebugOutput {
    const char* name;      // Display name
    const char* category;  // Category for grouping (e.g., "GBuffer", "Lighting")
    PtsTexture texture;    // Texture handle
    uint32_t mip_level;    // Which mip to display
    uint32_t array_slice;  // Which array slice to display
} PtsDebugOutput;

// ============================================================================
// Renderer Plugin Interface V1
// ============================================================================

typedef struct RendererPluginInterfaceV1 {
    // ------------------------------------------------------------------------
    // Required: Build render graph for current frame
    // ------------------------------------------------------------------------
    // Called each frame to construct the render graph. The plugin should:
    //   1. Import required textures from io->
    //   2. Create transient resources as needed
    //   3. Add passes to build the frame
    //
    // The graph is automatically compiled and executed after this returns.
    //
    void (*build_graph)(const PtsHostApi* host, const PtsFrameParams* frame,
                        const PtsViewParams* view, const PtsFrameIO* io);

    // ------------------------------------------------------------------------
    // Optional: Handle resize events
    // ------------------------------------------------------------------------
    // Called when the output resolution changes. Plugins should invalidate
    // resolution-dependent resources (e.g., G-buffers, history).
    // May be null if plugin doesn't need resize handling.
    //
    void (*on_resize)(uint32_t w, uint32_t h);

    // ------------------------------------------------------------------------
    // Optional: Expose debug outputs
    // ------------------------------------------------------------------------
    // Returns a list of intermediate textures for debug visualization.
    // list_blob receives PtsDebugOutput[] data.
    // May be null if plugin doesn't expose debug outputs.
    //
    void (*get_debug_outputs)(/*out*/ PtsSpan* list_blob);

    // ------------------------------------------------------------------------
    // Optional: Settings management
    // ------------------------------------------------------------------------
    // set_settings_blob: Apply settings from a blob (host-defined format).
    // get_settings_schema: Return schema describing available settings.
    // May be null if plugin doesn't have configurable settings.
    //
    void (*set_settings_blob)(PtsSpan blob);
    void (*get_settings_schema)(/*out*/ PtsSpan* schema_blob);

} RendererPluginInterfaceV1;
