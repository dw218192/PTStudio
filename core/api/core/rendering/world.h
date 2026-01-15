#pragma once
#include <stdint.h>

#include "../types.h"
#include "types.h"

// ============================================================================
// Render World API
// ============================================================================
//
// Provides read-only access to scene data prepared by the host for rendering.
// The world contains named buffers with structured data (instances, materials,
// lights, acceleration structures, etc.).
//
// Lifetime:
//   - Buffer handles returned by get_buffer() are valid for the current frame.
//   - PtsSpan returned by get_buffer_layout() is valid until the next call to
//     get_buffer_layout() or until the world is destroyed.
//   - Layout data should be cached by plugins; use schema_version() to detect
//     changes that require re-caching.
//
// Threading:
//   - World API is read-only and thread-safe for concurrent access.
//

// ============================================================================
// Buffer Layout Description
// ============================================================================

typedef enum PtsLayoutFieldType : uint32_t {
    PTS_FIELD_FLOAT = 1,
    PTS_FIELD_FLOAT2 = 2,
    PTS_FIELD_FLOAT3 = 3,
    PTS_FIELD_FLOAT4 = 4,
    PTS_FIELD_INT = 5,
    PTS_FIELD_INT2 = 6,
    PTS_FIELD_INT3 = 7,
    PTS_FIELD_INT4 = 8,
    PTS_FIELD_UINT = 9,
    PTS_FIELD_UINT2 = 10,
    PTS_FIELD_UINT3 = 11,
    PTS_FIELD_UINT4 = 12,
    PTS_FIELD_MAT3X3 = 13,
    PTS_FIELD_MAT4X4 = 14,
} PtsLayoutFieldType;

typedef struct PtsLayoutField {
    const char* name;              // Field name (lifetime: until schema changes)
    uint32_t offset_bytes;         // Offset within struct
    PtsLayoutFieldType type;       // Field type
} PtsLayoutField;

typedef struct PtsBufferLayout {
    uint32_t stride_bytes;         // Size of one element
    uint32_t field_count;          // Number of fields
    const PtsLayoutField* fields;  // Array of field descriptors
} PtsBufferLayout;

// ============================================================================
// Render World API
// ============================================================================

typedef struct PtsRenderWorldApi {
    // ------------------------------------------------------------------------
    // Buffer access
    // ------------------------------------------------------------------------
    // Get a named buffer from the world. Common names include:
    //   - "Instances"  : Instance transforms and material indices
    //   - "Materials"  : Material properties
    //   - "Lights"     : Light data
    //   - "TLAS"       : Top-level acceleration structure (for RT)
    //
    // Returns null handle ({0}) if buffer doesn't exist.
    PtsBuffer (*get_buffer)(const char* name);

    // ------------------------------------------------------------------------
    // Layout introspection
    // ------------------------------------------------------------------------
    // Returns layout description for a named buffer.
    // The returned PtsSpan contains PtsBufferLayout data.
    // Returns empty span if buffer doesn't exist.
    PtsSpan (*get_buffer_layout)(const char* name);

    // ------------------------------------------------------------------------
    // Schema versioning
    // ------------------------------------------------------------------------
    // Returns a version number that changes when buffer layouts change.
    // Plugins should cache layouts and re-query when version changes.
    uint64_t (*schema_version)(void);

    // ------------------------------------------------------------------------
    // Buffer queries
    // ------------------------------------------------------------------------
    // Get the number of elements in a buffer.
    uint64_t (*get_buffer_element_count)(const char* name);

} PtsRenderWorldApi;
