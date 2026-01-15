#pragma once
#include <stdint.h>

#include "../types.h"
#include "types.h"

typedef struct PtsLayoutField {
    const char* name;
    uint32_t offset_bytes;
    uint32_t type;  // define your own type enum (float3, u32, etc.)
} PtsLayoutField;

typedef struct PtsBufferLayout {
    uint32_t stride_bytes;
    uint32_t field_count;
    const PtsLayoutField* fields;
} PtsBufferLayout;

typedef struct PtsRenderWorldApi {
    // Named tables: "Instances", "Materials", "Lights", "TLAS", etc.
    PtsBuffer (*get_buffer)(const char* name);
    PtsSpan (*get_buffer_layout)(
        const char* name);  // returns PtsBufferLayout bytes, versioned blob
    uint64_t (*schema_version)(void);
} PtsRenderWorldApi;