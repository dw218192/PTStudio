#pragma once
#include <stdint.h>

#include "types.h"

// ============================================================================
// Lifetime & Ownership Conventions
// ============================================================================
//
// String parameters (const char*):
//   - All string parameters (debug_name, pass name, blackboard keys) must remain
//     valid until end() is called on the current graph.
//   - The host does NOT copy strings; callers must ensure lifetime.
//
// Span returns (PtsSpan):
//   - Spans returned by get_* functions are valid until the next call to the
//     same function, or until the owning object is destroyed.
//
// Transient resources (created via create_*):
//   - Valid only for the current frame (between begin() and end()).
//   - Handles become invalid after end() is called.
//
// Persistent resources (created via get_or_create_persistent_*):
//   - Valid across frames until the plugin is destroyed.
//   - Must be imported into each frame's graph before use in passes.
//
// Imported resources:
//   - Caller retains ownership; the graph only references them.
//
// ============================================================================

// ============================================================================
// Error Handling
// ============================================================================
//
// Functions that can fail return null handles ({0}) on failure.
// Use get_last_error() to retrieve the error code after a null return.
// Error state is per-graph and cleared on each successful operation.
//

typedef enum PtsGraphError : uint32_t {
    PTS_GRAPH_OK = 0,
    PTS_GRAPH_ERR_INVALID_HANDLE = 1,
    PTS_GRAPH_ERR_INVALID_DESC = 2,
    PTS_GRAPH_ERR_OUT_OF_MEMORY = 3,
    PTS_GRAPH_ERR_FORMAT_NOT_SUPPORTED = 4,
    PTS_GRAPH_ERR_RESOURCE_LIMIT = 5,
    PTS_GRAPH_ERR_INVALID_STATE = 6,
    PTS_GRAPH_ERR_NOT_FOUND = 7,
} PtsGraphError;

// ============================================================================
// Formats
// ============================================================================

typedef enum PtsFormat : uint32_t {
    PTS_FMT_UNKNOWN = 0,
    // 8-bit
    PTS_FMT_R8_UNORM,
    PTS_FMT_RG8_UNORM,
    PTS_FMT_RGBA8_UNORM,
    PTS_FMT_RGBA8_SRGB,
    // 16-bit float
    PTS_FMT_R16F,
    PTS_FMT_RG16F,
    PTS_FMT_RGBA16F,
    // 32-bit float
    PTS_FMT_R32F,
    PTS_FMT_RG32F,
    PTS_FMT_RGBA32F,
    // 32-bit uint/int
    PTS_FMT_R32_UINT,
    PTS_FMT_RG32_UINT,
    PTS_FMT_RGBA32_UINT,
    // Depth/stencil
    PTS_FMT_D16_UNORM,
    PTS_FMT_D32F,
    PTS_FMT_D24_UNORM_S8_UINT,
    PTS_FMT_D32F_S8_UINT,
} PtsFormat;

// ============================================================================
// Texture Types
// ============================================================================

typedef enum PtsTexUsage : uint32_t {
    PTS_TEX_USAGE_SRV = 1u << 0,
    PTS_TEX_USAGE_UAV = 1u << 1,
    PTS_TEX_USAGE_RTV = 1u << 2,
    PTS_TEX_USAGE_DSV = 1u << 3,
    PTS_TEX_USAGE_COPY_SRC = 1u << 4,
    PTS_TEX_USAGE_COPY_DST = 1u << 5,
} PtsTexUsage;

typedef struct PtsSubresourceRange {
    uint16_t base_mip;
    uint16_t mip_count;
    uint16_t base_layer;
    uint16_t layer_count;
} PtsSubresourceRange;

typedef enum PtsViewType : uint32_t {
    PTS_VIEW_SRV = 1,
    PTS_VIEW_UAV = 2,
    PTS_VIEW_RTV = 3,
    PTS_VIEW_DSV = 4,
} PtsViewType;

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
    PtsViewType view_type;
    PtsSubresourceRange range;
} PtsTextureViewDesc;

// ============================================================================
// Buffer Types
// ============================================================================

typedef enum PtsBufUsage : uint32_t {
    PTS_BUF_USAGE_VERTEX = 1u << 0,
    PTS_BUF_USAGE_INDEX = 1u << 1,
    PTS_BUF_USAGE_CONSTANT = 1u << 2,
    PTS_BUF_USAGE_SRV = 1u << 3,
    PTS_BUF_USAGE_UAV = 1u << 4,
    PTS_BUF_USAGE_INDIRECT = 1u << 5,
    PTS_BUF_USAGE_COPY_SRC = 1u << 6,
    PTS_BUF_USAGE_COPY_DST = 1u << 7,
} PtsBufUsage;

typedef struct PtsBufferDesc {
    uint64_t size_bytes;
    uint32_t usage_bits;   // PtsBufUsage
    uint32_t struct_stride; // 0 for raw buffers, >0 for structured
} PtsBufferDesc;

typedef struct PtsBufferViewDesc {
    PtsBuffer buf;
    uint64_t offset_bytes;
    uint64_t size_bytes;    // 0 = rest of buffer
    PtsFormat format;       // For typed buffers; PTS_FMT_UNKNOWN for structured/raw
    uint32_t struct_stride; // For structured buffer views
} PtsBufferViewDesc;

// ============================================================================
// Sampler Types
// ============================================================================

typedef enum PtsFilter : uint32_t {
    PTS_FILTER_NEAREST = 0,
    PTS_FILTER_LINEAR = 1,
} PtsFilter;

typedef enum PtsAddressMode : uint32_t {
    PTS_ADDRESS_WRAP = 0,
    PTS_ADDRESS_CLAMP = 1,
    PTS_ADDRESS_MIRROR = 2,
    PTS_ADDRESS_BORDER = 3,
} PtsAddressMode;

typedef enum PtsCompareOp : uint32_t {
    PTS_COMPARE_NONE = 0,
    PTS_COMPARE_LESS = 1,
    PTS_COMPARE_LESS_EQUAL = 2,
    PTS_COMPARE_GREATER = 3,
    PTS_COMPARE_GREATER_EQUAL = 4,
    PTS_COMPARE_EQUAL = 5,
    PTS_COMPARE_NOT_EQUAL = 6,
    PTS_COMPARE_ALWAYS = 7,
    PTS_COMPARE_NEVER = 8,
} PtsCompareOp;

typedef struct PtsSamplerDesc {
    PtsFilter min_filter;
    PtsFilter mag_filter;
    PtsFilter mip_filter;
    PtsAddressMode address_u;
    PtsAddressMode address_v;
    PtsAddressMode address_w;
    float mip_lod_bias;
    float max_anisotropy;  // 1.0 = disabled
    PtsCompareOp compare_op;
    float min_lod;
    float max_lod;
    float border_color[4];
} PtsSamplerDesc;

// ============================================================================
// Pipeline Types
// ============================================================================

typedef enum PtsPipelineType : uint32_t {
    PTS_PIPELINE_GRAPHICS = 1,
    PTS_PIPELINE_COMPUTE = 2,
} PtsPipelineType;

typedef enum PtsPrimitiveTopology : uint32_t {
    PTS_TOPOLOGY_TRIANGLE_LIST = 0,
    PTS_TOPOLOGY_TRIANGLE_STRIP = 1,
    PTS_TOPOLOGY_LINE_LIST = 2,
    PTS_TOPOLOGY_LINE_STRIP = 3,
    PTS_TOPOLOGY_POINT_LIST = 4,
} PtsPrimitiveTopology;

typedef enum PtsCullMode : uint32_t {
    PTS_CULL_NONE = 0,
    PTS_CULL_FRONT = 1,
    PTS_CULL_BACK = 2,
} PtsCullMode;

typedef enum PtsFrontFace : uint32_t {
    PTS_FRONT_CCW = 0,
    PTS_FRONT_CW = 1,
} PtsFrontFace;

typedef enum PtsBlendFactor : uint32_t {
    PTS_BLEND_ZERO = 0,
    PTS_BLEND_ONE = 1,
    PTS_BLEND_SRC_COLOR = 2,
    PTS_BLEND_INV_SRC_COLOR = 3,
    PTS_BLEND_SRC_ALPHA = 4,
    PTS_BLEND_INV_SRC_ALPHA = 5,
    PTS_BLEND_DST_COLOR = 6,
    PTS_BLEND_INV_DST_COLOR = 7,
    PTS_BLEND_DST_ALPHA = 8,
    PTS_BLEND_INV_DST_ALPHA = 9,
} PtsBlendFactor;

typedef enum PtsBlendOp : uint32_t {
    PTS_BLEND_OP_ADD = 0,
    PTS_BLEND_OP_SUBTRACT = 1,
    PTS_BLEND_OP_REV_SUBTRACT = 2,
    PTS_BLEND_OP_MIN = 3,
    PTS_BLEND_OP_MAX = 4,
} PtsBlendOp;

typedef struct PtsBlendState {
    uint8_t blend_enable;
    uint8_t _pad[3];
    PtsBlendFactor src_color;
    PtsBlendFactor dst_color;
    PtsBlendOp color_op;
    PtsBlendFactor src_alpha;
    PtsBlendFactor dst_alpha;
    PtsBlendOp alpha_op;
    uint8_t write_mask;  // RGBA bits
    uint8_t _pad2[3];
} PtsBlendState;

typedef struct PtsDepthStencilState {
    uint8_t depth_test_enable;
    uint8_t depth_write_enable;
    uint8_t stencil_enable;
    uint8_t _pad;
    PtsCompareOp depth_compare;
    // Stencil ops could be expanded here
} PtsDepthStencilState;

typedef struct PtsRasterizerState {
    PtsCullMode cull_mode;
    PtsFrontFace front_face;
    uint8_t depth_bias_enable;
    uint8_t wireframe;
    uint8_t _pad[2];
    float depth_bias_constant;
    float depth_bias_slope;
    float depth_bias_clamp;
} PtsRasterizerState;

typedef struct PtsVertexAttribute {
    uint32_t location;
    uint32_t binding;
    PtsFormat format;
    uint32_t offset;
} PtsVertexAttribute;

typedef struct PtsVertexBinding {
    uint32_t binding;
    uint32_t stride;
    uint8_t per_instance;  // 0 = per vertex, 1 = per instance
    uint8_t _pad[3];
} PtsVertexBinding;

typedef struct PtsVertexInputState {
    const PtsVertexAttribute* attributes;
    uint32_t attribute_count;
    const PtsVertexBinding* bindings;
    uint32_t binding_count;
} PtsVertexInputState;

typedef struct PtsShaderStage {
    const void* bytecode;
    uint64_t bytecode_size;
    const char* entry_point;
} PtsShaderStage;

typedef struct PtsGraphicsPipelineDesc {
    PtsShaderStage vertex_shader;
    PtsShaderStage pixel_shader;
    PtsVertexInputState vertex_input;
    PtsPrimitiveTopology topology;
    PtsRasterizerState rasterizer;
    PtsDepthStencilState depth_stencil;
    const PtsBlendState* blend_states;  // One per render target
    uint32_t blend_state_count;
    const PtsFormat* render_target_formats;
    uint32_t render_target_count;
    PtsFormat depth_stencil_format;  // PTS_FMT_UNKNOWN if none
} PtsGraphicsPipelineDesc;

typedef struct PtsComputePipelineDesc {
    PtsShaderStage compute_shader;
} PtsComputePipelineDesc;

// ============================================================================
// Attachment & Pass Types
// ============================================================================

typedef enum PtsStoreOp : uint8_t {
    PTS_STORE_STORE = 0,
    PTS_STORE_DONTCARE = 1
} PtsStoreOp;

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
    uint8_t _pad[2];
    PtsClearValue clear;
} PtsAttachment;

// ============================================================================
// Resource Access Declaration (for scheduling/barriers)
// ============================================================================

typedef enum PtsAccess : uint32_t {
    PTS_ACCESS_SRV = 1,
    PTS_ACCESS_UAV_READ = 2,
    PTS_ACCESS_UAV_WRITE = 3,
    PTS_ACCESS_UAV_READ_WRITE = 4,
    PTS_ACCESS_RTV = 5,
    PTS_ACCESS_DSV_READ = 6,
    PTS_ACCESS_DSV_WRITE = 7,
    PTS_ACCESS_COPY_SRC = 8,
    PTS_ACCESS_COPY_DST = 9,
    PTS_ACCESS_VERTEX_BUFFER = 10,
    PTS_ACCESS_INDEX_BUFFER = 11,
    PTS_ACCESS_INDIRECT_ARGS = 12,
    PTS_ACCESS_CONSTANT_BUFFER = 13,
} PtsAccess;

typedef enum PtsResKind : uint32_t {
    PTS_RES_TEXTURE_VIEW = 1,
    PTS_RES_BUFFER_VIEW = 2,
} PtsResKind;

typedef struct PtsResUse {
    PtsResKind kind;
    PtsAccess access;
    union {
        PtsTexView tex_view;
        PtsBufView buf_view;
    };
} PtsResUse;

// Helper macros for PtsResUse initialization
#define PTS_TEX_USE(view, acc) \
    (PtsResUse) { .kind = PTS_RES_TEXTURE_VIEW, .access = (acc), .tex_view = (view) }
#define PTS_BUF_USE(view, acc) \
    (PtsResUse) { .kind = PTS_RES_BUFFER_VIEW, .access = (acc), .buf_view = (view) }

// ============================================================================
// Pass Types
// ============================================================================

// Forward decl for cmd encoder (see cmd.h for full API)
typedef struct PtsCmd PtsCmd;

// Pass callback type
typedef void (*PtsEncodeFn)(PtsCmd* cmd, void* user);

typedef struct PtsPassDesc {
    const char* name;  // Must remain valid until end() (see lifetime docs)

    const PtsResUse* resources;
    uint32_t resource_count;

    // For raster passes (optional; can be null/0 for compute/RT)
    const PtsAttachment* color_attachments;
    uint32_t color_count;
    const PtsAttachment* depth_attachment;  // null or pointer to single attachment

    // Execute callback
    PtsEncodeFn encode;
    void* user;
} PtsPassDesc;

// ============================================================================
// Render Graph API
// ============================================================================

typedef struct PtsRenderGraphApi {
    // ------------------------------------------------------------------------
    // Error handling
    // ------------------------------------------------------------------------
    // Returns the last error for the given graph. Cleared on successful operations.
    PtsGraphError (*get_last_error)(PtsGraph g);
    const char* (*get_error_message)(PtsGraphError err);

    // ------------------------------------------------------------------------
    // Graph lifecycle (typically per-frame)
    // ------------------------------------------------------------------------
    // Returns null handle ({0}) on failure; check get_last_error(null) for reason.
    PtsGraph (*begin)(void);
    void (*end)(PtsGraph g);

    // ------------------------------------------------------------------------
    // Transient texture resources
    // ------------------------------------------------------------------------
    // Returns null handle ({0}) on failure.
    PtsTexture (*create_texture)(PtsGraph g, const PtsTextureDesc* desc, const char* debug_name);
    PtsTexView (*create_tex_view)(PtsGraph g, const PtsTextureViewDesc* desc,
                                  const char* debug_name);

    // Import an externally-owned texture (swapchain/output/history/etc.)
    PtsTexture (*import_texture)(PtsGraph g, PtsTexture external_tex, const char* debug_name);

    // ------------------------------------------------------------------------
    // Transient buffer resources
    // ------------------------------------------------------------------------
    // Returns null handle ({0}) on failure.
    PtsBuffer (*create_buffer)(PtsGraph g, const PtsBufferDesc* desc, const char* debug_name);
    PtsBufView (*create_buf_view)(PtsGraph g, const PtsBufferViewDesc* desc,
                                  const char* debug_name);

    // Import an externally-owned buffer
    PtsBuffer (*import_buffer)(PtsGraph g, PtsBuffer external_buf, const char* debug_name);

    // ------------------------------------------------------------------------
    // Samplers (typically cached/deduplicated by host)
    // ------------------------------------------------------------------------
    // Returns null handle ({0}) on failure.
    PtsSampler (*create_sampler)(PtsGraph g, const PtsSamplerDesc* desc);

    // ------------------------------------------------------------------------
    // Pipelines
    // ------------------------------------------------------------------------
    // Returns null handle ({0}) on failure. Pipelines are cached by host.
    PtsPipeline (*create_graphics_pipeline)(PtsGraph g, const PtsGraphicsPipelineDesc* desc,
                                            const char* debug_name);
    PtsPipeline (*create_compute_pipeline)(PtsGraph g, const PtsComputePipelineDesc* desc,
                                           const char* debug_name);

    // ------------------------------------------------------------------------
    // Pass creation
    // ------------------------------------------------------------------------
    // Returns null handle ({0}) on failure.
    PtsPass (*add_pass)(PtsGraph g, const PtsPassDesc* desc);

    // ------------------------------------------------------------------------
    // Blackboard (share handles between passes without static globals)
    // ------------------------------------------------------------------------
    // Keys must remain valid until end() (see lifetime docs).
    void (*bb_set_u64)(PtsGraph g, const char* key, uint64_t v);
    uint64_t (*bb_get_u64)(PtsGraph g, const char* key, uint64_t default_v);
    void (*bb_set_tex)(PtsGraph g, const char* key, PtsTexture v);
    PtsTexture (*bb_get_tex)(PtsGraph g, const char* key);
    void (*bb_set_buf)(PtsGraph g, const char* key, PtsBuffer v);
    PtsBuffer (*bb_get_buf)(PtsGraph g, const char* key);

    // ------------------------------------------------------------------------
    // Persistent resources (survive across frames, owned by host)
    // ------------------------------------------------------------------------
    // Keyed by string; host keeps them alive across frames and destroys on
    // plugin unload. Returned handle must be imported into graph before use.
    // Returns null handle ({0}) on failure.
    PtsTexture (*get_or_create_persistent_texture)(const char* key, const PtsTextureDesc* desc,
                                                   const char* debug_name);
    PtsBuffer (*get_or_create_persistent_buffer)(const char* key, const PtsBufferDesc* desc,
                                                 const char* debug_name);
} PtsRenderGraphApi;

