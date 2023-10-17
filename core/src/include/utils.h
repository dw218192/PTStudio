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

// c++23 expected
#include <tl/expected.hpp>

// convenience helpers
#define TL_ERROR(msg) tl::unexpected { std::string{__FILE__} + ":" + std::to_string(__LINE__) + ": " + (msg) }
#define TL_GL_ERROR(err) TL_ERROR(reinterpret_cast<char const*>(glewGetErrorString(err)))

#define CHECK_GL_ERROR() do { \
	auto err = glGetError(); \
	if (err != GL_NO_ERROR) return TL_ERROR(reinterpret_cast<char const*>(glewGetErrorString(err))); \
} while (0)
#define TL_CHECK(func_call) do { \
	auto res = func_call;\
	if (!res) return TL_ERROR(res.error()); \
} while (0)
#define TL_CHECK_FWD(func_call) do { \
	auto res = func_call;\
	if (!res) return res; \
} while (0)
#define TL_CHECK_RET(func_call) ({ \
	auto res = func_call; \
	if (!res) return TL_ERROR(res.error()); \
	res.value(); \
})
#define TL_CHECK_RET_FWD(func_call) ({ \
	auto res = func_call; \
	if (!res) return res; \
	res.value(); \
})