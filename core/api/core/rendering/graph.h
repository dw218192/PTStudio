#pragma once
#include <stdint.h>

#include "types.h"

typedef enum PtsFormat : uint32_t {
    PTS_FMT_UNKNOWN = 0,
    PTS_FMT_RGBA8_UNORM,
    PTS_FMT_RGBA8_SRGB,
    PTS_FMT_RGBA16F,
    PTS_FMT_R16F,
    PTS_FMT_R32F,
    PTS_FMT_D32F,
} PtsFormat;

typedef enum PtsTexUsage : uint32_t {
    PTS_TEX_USAGE_SRV = 1u << 0,
    PTS_TEX_USAGE_UAV = 1u << 1,
    PTS_TEX_USAGE_RTV = 1u << 2,
    PTS_TEX_USAGE_DSV = 1u << 3,
    PTS_TEX_USAGE_COPY = 1u << 4,
} PtsTexUsage;

typedef struct PtsSubresourceRange {
    uint16_t base_mip;
    uint16_t mip_count;
    uint16_t base_layer;
    uint16_t layer_count;
} PtsSubresourceRange;

typedef enum PtsViewUsage : uint32_t {
    PTS_VIEW_SRV = 1,
    PTS_VIEW_UAV = 2,
    PTS_VIEW_RTV = 3,
    PTS_VIEW_DSV = 4,
} PtsViewUsage;

typedef struct PtsTextureDesc {
    uint32_t w, h;
    uint16_t mips;
    uint16_t layers;
    PtsFormat format;
    uint32_t usage_bits;  // PtsTexUsage
} PtsTextureDesc;

typedef struct PtsTextureViewDesc {
    PtsTexture tex;
    PtsFormat view_format;  // allow reinterpret (e.g. SRGB/UNORM)
    PtsViewUsage usage;
    PtsSubresourceRange range;
} PtsTextureViewDesc;

typedef enum PtsStoreOp : uint8_t { PTS_STORE_STORE = 0, PTS_STORE_DONTCARE = 1 } PtsStoreOp;

typedef enum PtsLoadOp : uint8_t {
    PTS_LOAD_LOAD = 0,
    PTS_LOAD_CLEAR = 1,
    PTS_LOAD_DONTCARE = 2
} PtsLoadOp;

typedef struct PtsClearValue {
    float rgba[4];
    float depth;
    uint32_t stencil;
} PtsClearValue;

typedef struct PtsAttachment {
    PtsTexView view;
    PtsLoadOp load_op;
    PtsStoreOp store_op;
    PtsClearValue clear;
} PtsAttachment;

// Resource access declaration (for scheduling/barriers).
typedef enum PtsAccess : uint32_t {
    PTS_ACCESS_SRV = 1,
    PTS_ACCESS_UAV = 2,
    PTS_ACCESS_RTV = 3,
    PTS_ACCESS_DSV = 4,
    PTS_ACCESS_COPY_SRC = 5,
    PTS_ACCESS_COPY_DST = 6,
} PtsAccess;

typedef struct PtsResUse {
    // either a texture view or buffer view; keep v1 simple: one union.
    uint64_t view_handle;  // PtsTexView.h or PtsBufView.h
    uint32_t access;       // PtsAccess
    uint32_t _pad;
} PtsResUse;

// Forward decl for cmd encoder
typedef struct PtsCmd PtsCmd;

// Pass callback types
typedef void (*PtsEncodeFn)(PtsCmd* cmd, void* user);
typedef void (*PtsExternalFn)(void* user, const void* interop_api /*PtsInteropApi*/);

typedef enum PtsPassKind : uint32_t { PTS_PASS_ENCODED = 1, PTS_PASS_EXTERNAL = 2 } PtsPassKind;

typedef struct PtsPassDesc {
    const char* name;
    PtsPassKind kind;

    const PtsResUse* reads;
    uint32_t read_count;
    const PtsResUse* writes;
    uint32_t write_count;

    // For raster-like passes (optional; can be empty for compute/rt)
    const PtsAttachment* color_attachments;
    uint32_t color_count;
    const PtsAttachment* depth_attachment;  // 0 or 1

    // Execute callback
    union {
        PtsEncodeFn encode;
        PtsExternalFn external;
    } fn;
    void* user;
} PtsPassDesc;

typedef struct PtsRenderGraphApi {
    // Graph lifecycle (typically per-frame)
    PtsGraph (*begin)(void);
    void (*end)(PtsGraph g);

    // Transient resources
    PtsTexture (*create_texture)(PtsGraph g, const PtsTextureDesc* desc, const char* debug_name);
    PtsTexView (*create_tex_view)(PtsGraph g, const PtsTextureViewDesc* desc,
                                  const char* debug_name);

    // Import an externally-owned texture (swapchain/output/history/etc.)
    PtsTexture (*import_texture)(PtsGraph g, PtsTexture tex, const char* debug_name);

    // Pass creation
    PtsPass (*add_pass)(PtsGraph g, const PtsPassDesc* desc);

    // Blackboard (share handles between passes without static globals)
    void (*bb_set_u64)(PtsGraph g, const char* key, uint64_t v);
    uint64_t (*bb_get_u64)(PtsGraph g, const char* key, uint64_t default_v);

    // Persistent cache: plugin-owned persistent resources keyed by
    // string Host keeps them alive across frames and destroys on plugin destroy.
    PtsTexture (*get_or_create_persistent_texture)(const char* key, const PtsTextureDesc* desc,
                                                   const char* debug_name);
} PtsRenderGraphApi;
