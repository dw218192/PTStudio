#pragma once

#ifdef __CUDACC__
#include <device_launch_parameters.h>
#define DEVICE __device__
#define HOST __host__
#define INLINE __forceinline__
#define GLOBAL __global__
#define NODISCARD
#define HOST_DEVICE __host__ __device__
#else
#define DEVICE
#define HOST
#define INLINE
#define GLOBAL
#define NODISCARD [[nodiscard]]
#define HOST_DEVICE
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