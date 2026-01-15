#pragma once
#include <stdint.h>

#include "graph.h"
#include "types.h"

// ============================================================================
// Command Encoder API
// ============================================================================
//
// PtsCmd is the command encoder passed to pass callbacks. It provides methods
// to record GPU commands within a pass.
//
// Lifetime:
//   - PtsCmd* is only valid within the scope of the pass callback.
//   - Do not store or use PtsCmd* outside the callback.
//
// Threading:
//   - Each pass callback receives its own PtsCmd*.
//   - Do not share PtsCmd* between threads.
//

// ============================================================================
// Draw/Dispatch Arguments
// ============================================================================

typedef struct PtsDrawArgs {
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
} PtsDrawArgs;

typedef struct PtsDrawIndexedArgs {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t first_instance;
} PtsDrawIndexedArgs;

typedef struct PtsDispatchArgs {
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
} PtsDispatchArgs;

// ============================================================================
// Viewport & Scissor
// ============================================================================

typedef struct PtsViewport {
    float x, y;
    float width, height;
    float min_depth, max_depth;
} PtsViewport;

typedef struct PtsScissor {
    int32_t x, y;
    uint32_t width, height;
} PtsScissor;

// ============================================================================
// Resource Binding
// ============================================================================

typedef enum PtsBindPoint : uint32_t {
    PTS_BIND_GRAPHICS = 0,
    PTS_BIND_COMPUTE = 1,
} PtsBindPoint;

// Descriptor binding - supports both table-based and direct binding
typedef struct PtsDescriptorBinding {
    uint32_t set;      // Descriptor set index
    uint32_t binding;  // Binding within set
    union {
        PtsTexView tex_view;
        PtsBufView buf_view;
        PtsSampler sampler;
        struct {
            PtsBuffer buf;
            uint64_t offset;
            uint64_t size;  // 0 = rest of buffer
        } cbv;
    };
    enum {
        PTS_DESC_TEX_VIEW,
        PTS_DESC_BUF_VIEW,
        PTS_DESC_SAMPLER,
        PTS_DESC_CBV,
    } type;
} PtsDescriptorBinding;

// ============================================================================
// Copy Operations
// ============================================================================

typedef struct PtsTextureCopyRegion {
    PtsSubresourceRange src_subresource;
    uint32_t src_offset[3];  // x, y, z
    PtsSubresourceRange dst_subresource;
    uint32_t dst_offset[3];  // x, y, z
    uint32_t extent[3];      // width, height, depth
} PtsTextureCopyRegion;

typedef struct PtsBufferCopyRegion {
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t size;
} PtsBufferCopyRegion;

typedef struct PtsBufferTextureCopyRegion {
    uint64_t buffer_offset;
    uint32_t buffer_row_pitch;    // 0 = tightly packed
    uint32_t buffer_slice_pitch;  // 0 = tightly packed
    PtsSubresourceRange tex_subresource;
    uint32_t tex_offset[3];
    uint32_t tex_extent[3];
} PtsBufferTextureCopyRegion;

// ============================================================================
// Command Encoder Function Table
// ============================================================================

typedef struct PtsCmdApi {
    // ------------------------------------------------------------------------
    // Pipeline state
    // ------------------------------------------------------------------------
    void (*set_pipeline)(PtsCmd* cmd, PtsPipeline pipeline);

    // ------------------------------------------------------------------------
    // Viewport & scissor (for graphics passes)
    // ------------------------------------------------------------------------
    void (*set_viewports)(PtsCmd* cmd, const PtsViewport* viewports, uint32_t count);
    void (*set_scissors)(PtsCmd* cmd, const PtsScissor* scissors, uint32_t count);

    // ------------------------------------------------------------------------
    // Vertex & index buffers (for graphics passes)
    // ------------------------------------------------------------------------
    void (*set_vertex_buffers)(PtsCmd* cmd, uint32_t first_binding, const PtsBuffer* buffers,
                               const uint64_t* offsets, uint32_t count);
    void (*set_index_buffer)(PtsCmd* cmd, PtsBuffer buffer, uint64_t offset, PtsFormat format);

    // ------------------------------------------------------------------------
    // Resource binding
    // ------------------------------------------------------------------------
    // Bind individual descriptors
    void (*bind_descriptors)(PtsCmd* cmd, PtsBindPoint bind_point,
                             const PtsDescriptorBinding* bindings, uint32_t count);

    // Push constants (small, fast-path uniform data)
    void (*push_constants)(PtsCmd* cmd, PtsBindPoint bind_point, uint32_t offset, const void* data,
                           uint32_t size);

    // ------------------------------------------------------------------------
    // Draw commands (graphics passes only)
    // ------------------------------------------------------------------------
    void (*draw)(PtsCmd* cmd, const PtsDrawArgs* args);
    void (*draw_indexed)(PtsCmd* cmd, const PtsDrawIndexedArgs* args);
    void (*draw_indirect)(PtsCmd* cmd, PtsBuffer args_buffer, uint64_t offset, uint32_t draw_count,
                          uint32_t stride);
    void (*draw_indexed_indirect)(PtsCmd* cmd, PtsBuffer args_buffer, uint64_t offset,
                                  uint32_t draw_count, uint32_t stride);

    // ------------------------------------------------------------------------
    // Dispatch commands (compute passes only)
    // ------------------------------------------------------------------------
    void (*dispatch)(PtsCmd* cmd, const PtsDispatchArgs* args);
    void (*dispatch_indirect)(PtsCmd* cmd, PtsBuffer args_buffer, uint64_t offset);

    // ------------------------------------------------------------------------
    // Copy commands
    // ------------------------------------------------------------------------
    void (*copy_buffer)(PtsCmd* cmd, PtsBuffer src, PtsBuffer dst,
                        const PtsBufferCopyRegion* regions, uint32_t region_count);
    void (*copy_texture)(PtsCmd* cmd, PtsTexture src, PtsTexture dst,
                         const PtsTextureCopyRegion* regions, uint32_t region_count);
    void (*copy_buffer_to_texture)(PtsCmd* cmd, PtsBuffer src, PtsTexture dst,
                                   const PtsBufferTextureCopyRegion* regions,
                                   uint32_t region_count);
    void (*copy_texture_to_buffer)(PtsCmd* cmd, PtsTexture src, PtsBuffer dst,
                                   const PtsBufferTextureCopyRegion* regions,
                                   uint32_t region_count);

    // ------------------------------------------------------------------------
    // Debug markers
    // ------------------------------------------------------------------------
    void (*begin_debug_region)(PtsCmd* cmd, const char* name, const float color[4]);
    void (*end_debug_region)(PtsCmd* cmd);
    void (*insert_debug_marker)(PtsCmd* cmd, const char* name, const float color[4]);

    // ------------------------------------------------------------------------
    // Buffer updates (for small, infrequent updates)
    // ------------------------------------------------------------------------
    // Writes data to buffer. For large/frequent updates, use staging buffers.
    void (*update_buffer)(PtsCmd* cmd, PtsBuffer dst, uint64_t offset, const void* data,
                          uint64_t size);

} PtsCmdApi;

// ============================================================================
// PtsCmd Structure
// ============================================================================
//
// The actual PtsCmd structure contains a pointer to the API and internal state.
// Plugins use the API via: cmd->api->draw(cmd, &args);
//
// Convenience macros are provided below.
//

struct PtsCmd {
    const PtsCmdApi* api;
    void* _internal;  // Host-owned state, do not touch
};
