#pragma once

#include "device_launch_parameters.h"

#ifdef __CUDACC__
#define DEVICE __device__
#define HOST __host__
#define INLINE __forceinline__
#define GLOBAL __global__

#else
#define DEVICE
#define HOST
#define INLINE
#define GLOBAL
#endif


#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
