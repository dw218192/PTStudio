#pragma once
#include <stdint.h>

#include "../plugin.h"
#include "../types.h"
#include "types.h"

#define RENDERER_PLUGIN_INTERFACE_V1_ID "renderer.interface.v1"

// Keep this POD and stable. Add fields only at the end in future versions.
typedef struct PtsViewParams {
    float view[16];
    float proj[16];
    float view_proj[16];
    float prev_view_proj[16];
    float camera_pos[3];
    float _pad0;
    float jitter_xy[2];
    float dt_seconds;
    uint32_t viewport_w;
    uint32_t viewport_h;
} PtsViewParams;

typedef struct PtsFrameParams {
    uint64_t frame_index;
    double time_seconds;
} PtsFrameParams;

// Inputs/outputs are handles (swapchain image or host textures).
typedef struct PtsFrameIO {
    PtsTexture output;   // presentable or offscreen target chosen by host
    PtsTexture history;  // optional (0 if none)
    PtsTexture depth;    // optional
    PtsTexture motion;   // optional
    PtsTexture normal;   // optional
    PtsTexture albedo;   // optional
} PtsFrameIO;

typedef struct RendererPluginInterfaceV1 {
    void (*build_graph)(void* instance, const PtsHostApi* host, const PtsFrameParams* frame,
                        const PtsViewParams* view, const PtsFrameIO* io);
    // Optional callbacks
    void (*on_resize)(void* instance, uint32_t w, uint32_t h);
    void (*get_debug_outputs)(void* instance,
                              /*out*/ PtsSpan* list_blob);  // host-defined blob (names+handles)
    void (*set_settings_blob)(void* instance, PtsSpan blob);
    void (*get_settings_schema)(void* instance, /*out*/ PtsSpan* schema_blob);  // host-defined blob

} RendererPluginInterfaceV1;