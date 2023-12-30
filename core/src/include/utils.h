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

template <typename T>
constexpr T div_up(T x, T y) {
	return (x + y - 1) / y;
}

#include <sstream>
// c++23 expected, fmt
#include <tl/expected.hpp>
#include <fmt/core.h>

namespace PTS {
	template <typename Str, typename... Args>
	auto do_format(Str str, Args&&... args) noexcept {
		try {
			return fmt::format(str, std::forward<Args>(args)...);
		} catch (fmt::format_error const&) {
			std::ostringstream ss;
			ss << str << '\n';
			(ss << ... << std::forward<Args>(args));
			return ss.str();
		}
	}
} // namespace PTS

// convenience helpers
#ifdef NDEBUG
#define TL_ERROR(msg, ...) tl::unexpected { fmt::format(msg, __VA_ARGS__) }
#else
#define TL_ERROR(msg, ...) tl::unexpected { std::string{__FILE__} + ":" + std::to_string(__LINE__) + ": " + PTS::do_format(msg, __VA_ARGS__) }
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
#define TL_CHECK_AND_PASS(func_call) do { \
	auto res = func_call;\
	if (!res) return res; \
} while (0)
#define TL_TRY_ASSIGN(out, func_call) do { \
	auto res = func_call;\
	if (!res) return TL_ERROR(res.error()); \
	out = std::move(res.value());\
} while (0)
#define TL_TRY_ASSIGN_AND_PASS(out, func_call) do { \
	auto res = func_call;\
	if (!res) return res; \
	out = std::move(res.value());\
} while (0)


// non-fatal errors
#define CHECK_GL_ERROR_NON_FATAL(app, level) do { \
	auto err = glGetError(); \
	if (err != GL_NO_ERROR) (app).log(level, glewGetErrorString(err)); \
} while (0)

#define TL_CHECK_NON_FATAL(app, level, func_call) do { \
	auto res = func_call;\
	if (!res) (app)->log(level, res.error()); \
} while (0)

#define TL_TRY_ASSIGN_NON_FATAL(out, app, level, func_call) do { \
	auto res = func_call;\
	if (!res) (app)->log(level, res.error()); \
	else { out = std::move(res.value()); }\
} while (0)


// useful typedefs
// TODO: may implement these as classes later
// because raw ptrs are not zero default-init'ed
namespace PTS {
	// non-owning ptr
	template <typename T>
	using ObserverPtr = T*;

	// non-owning ptr to a const, essentially a view
	template <typename T>
	using ViewPtr = T const*;

	// non-owning view
	template <typename T>
	using Ref = std::reference_wrapper<T>;

	// non-owning view to a const
	template <typename T>
	using View = std::reference_wrapper<T const>;
}

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

/**
 * @brief Declares a static init method that will be called the first time the class is instantiated.
 * @param method_name The name of the static init method.
 * @note The static init method must be static and return void.
*/
#define DECL_DEFERRED_STATIC_INIT(method_name)\
	static auto method_name() -> void;\
	static inline int _static_init_if_not() {\
		static bool s_init = false;\
		if (!s_init) {\
			method_name();\
			s_init = true;\
		}\
		return 0;\
	}\
	int m_static_init_helper = [] { return _static_init_if_not(); }()
