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

#ifdef __INTELLISENSE__
#define KERN_PARAM(x,y)
#define __syncthreads()
#else
#define KERN_PARAM(x,y) <<< x,y >>>
#endif

template<typename T>
constexpr T div_up(T x, T y) {
    return (x + y - 1) / y;
}