#pragma once
#include <stdint.h>

// clang-format off
typedef struct { uint64_t h; } PtsTexture;
typedef struct { uint64_t h; } PtsBuffer;
typedef struct { uint64_t h; } PtsTexView;
typedef struct { uint64_t h; } PtsBufView;
typedef struct { uint64_t h; } PtsSampler;
typedef struct { uint64_t h; } PtsPipeline;
typedef struct { uint64_t h; } PtsPass;
typedef struct { uint64_t h; } PtsGraph;

// Null handle constants
#define PTS_NULL_TEXTURE  ((PtsTexture){0})
#define PTS_NULL_BUFFER   ((PtsBuffer){0})
#define PTS_NULL_TEXVIEW  ((PtsTexView){0})
#define PTS_NULL_BUFVIEW  ((PtsBufView){0})
#define PTS_NULL_SAMPLER  ((PtsSampler){0})
#define PTS_NULL_PIPELINE ((PtsPipeline){0})
#define PTS_NULL_PASS     ((PtsPass){0})
#define PTS_NULL_GRAPH    ((PtsGraph){0})

// Check if handle is null
#define PTS_IS_NULL(handle) ((handle).h == 0)
// clang-format on