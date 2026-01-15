#pragma once
#include <stdint.h>

typedef struct PtsSpan {
    const void* data;
    uint32_t size_bytes;
} PtsSpan;

typedef struct PtsMutSpan {
    void* data;
    uint32_t size_bytes;
} PtsMutSpan;