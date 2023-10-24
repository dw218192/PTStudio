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
#ifdef NDEBUG
#define TL_ERROR(msg) tl::unexpected { msg }
#else
#define TL_ERROR(msg) tl::unexpected { std::string{__FILE__} + ":" + std::to_string(__LINE__) + ": " + (msg) }
#endif


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
#define TL_ASSIGN(out, func_call) do { \
	auto res = func_call;\
	if (!res) return TL_ERROR(res.error()); \
	out = std::move(res.value());\
} while (0)
#define TL_ASSIGN_FWD(out, func_call) do { \
	auto res = func_call;\
	if (!res) return res; \
	out = std::move(res.value());\
} while (0)

// useful typedefs
// TODO: may implement these as classes later
// because raw ptrs are not zero default-init'ed

// non-owning ptr
template<typename T> using ObserverPtr = T*;

// non-owning ptr to a const, essentially a view
template<typename T> using ViewPtr = T const*;

// non-owing view to a const
template<typename T> using View = T const&;

#define NO_COPY_MOVE(Ty)\
Ty(Ty const&) = delete;\
Ty& operator=(Ty const&) = delete;\
Ty(Ty&&) = delete;\
Ty& operator=(Ty&&) = delete

#define DEFAULT_COPY_MOVE(Ty)\
Ty(Ty const&) = default;\
Ty& operator=(Ty const&) = default;\
Ty(Ty&&) = default;\
Ty& operator=(Ty&&) = default

#define NO_COPY(Ty)\
Ty(Ty const&) = delete;\
Ty& operator=(Ty const&) = delete

#define DECL_ENUM(name, ...)\
	enum class name { __VA_ARGS__, __COUNT }