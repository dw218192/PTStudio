#pragma once

#include <boost/assert.hpp>
#include <cstdio>   // std::fputs, std::fprintf
#include <cstdlib>  // std::abort
#include <cstring>  // std::strrchr (optional)
#include <string_view>

#if !defined(PTS_DIAG_ENABLE_STACKTRACE)
#define PTS_DIAG_ENABLE_STACKTRACE 1
#endif

#if PTS_DIAG_ENABLE_STACKTRACE
#include <boost/stacktrace/stacktrace.hpp>
#include <boost/stacktrace/stacktrace_fwd.hpp>
#include <sstream>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace pts::diagnostics {

inline void write_stderr(std::string_view s) noexcept {
    std::fwrite(s.data(), 1, s.size(), stderr);
}

inline void write_line(std::string_view s) noexcept {
    write_stderr(s);
    write_stderr("\n");
}

#if PTS_DIAG_ENABLE_STACKTRACE
inline std::string stacktrace_string() {
    std::ostringstream oss;
    oss << boost::stacktrace::stacktrace();
    return oss.str();
}
#endif

[[noreturn]] inline void trap() noexcept {
#if defined(_MSC_VER)
    __debugbreak();
    std::abort();  // fallback if continuing
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_trap();
#else
    std::abort();
#endif
}

[[noreturn]] inline void fail_fast(const char* kind, const char* expr, const char* file, int line,
                                   const char* msg = nullptr) noexcept {
    std::fprintf(stderr, "%s failed: %s\n  at %s:%d\n", kind, expr ? expr : "(none)", file, line);
    if (msg && *msg) {
        std::fprintf(stderr, "  message: %s\n", msg);
    }
#if PTS_DIAG_ENABLE_STACKTRACE
    // Best-effort: this might allocate; only on failure path.
    try {
        auto st = stacktrace_string();
        std::fputs("  stacktrace:\n", stderr);
        std::fputs(st.c_str(), stderr);
    } catch (...) {
        std::fputs("  stacktrace: <unavailable>\n", stderr);
    }
#endif
    trap();
}

}  // namespace pts::diagnostics

// --- Public macros ---
//
// Debug-only assert (Boost handles NDEBUG semantics)
#define ASSERT(x) BOOST_ASSERT(x)
#define ASSERT_MSG(x, m) BOOST_ASSERT_MSG(x, m)

// Always-on generic checks
#define CHECK(x)                                                                  \
    do {                                                                          \
        if (!(x)) ::pts::diagnostics::fail_fast("CHECK", #x, __FILE__, __LINE__); \
    } while (0)

#define CHECK_MSG(x, m)                                                                \
    do {                                                                               \
        if (!(x)) ::pts::diagnostics::fail_fast("CHECK", #x, __FILE__, __LINE__, (m)); \
    } while (0)

// Semantic checks (always-on) - use these for clearer intent
//
// PRECONDITION: caller provided valid inputs (function entry)
#define PRECONDITION(x)                                                                  \
    do {                                                                                 \
        if (!(x)) ::pts::diagnostics::fail_fast("PRECONDITION", #x, __FILE__, __LINE__); \
    } while (0)

#define PRECONDITION_MSG(x, m)                                                                \
    do {                                                                                      \
        if (!(x)) ::pts::diagnostics::fail_fast("PRECONDITION", #x, __FILE__, __LINE__, (m)); \
    } while (0)

// POSTCONDITION: outputs are valid (function exit)
#define POSTCONDITION(x)                                                                  \
    do {                                                                                  \
        if (!(x)) ::pts::diagnostics::fail_fast("POSTCONDITION", #x, __FILE__, __LINE__); \
    } while (0)

#define POSTCONDITION_MSG(x, m)                                                                \
    do {                                                                                       \
        if (!(x)) ::pts::diagnostics::fail_fast("POSTCONDITION", #x, __FILE__, __LINE__, (m)); \
    } while (0)

// INVARIANT: class/data structure invariants hold
#define INVARIANT(x)                                                                  \
    do {                                                                              \
        if (!(x)) ::pts::diagnostics::fail_fast("INVARIANT", #x, __FILE__, __LINE__); \
    } while (0)

#define INVARIANT_MSG(x, m)                                                                \
    do {                                                                                   \
        if (!(x)) ::pts::diagnostics::fail_fast("INVARIANT", #x, __FILE__, __LINE__, (m)); \
    } while (0)

#define PANIC(m)                                                                  \
    do {                                                                          \
        ::pts::diagnostics::fail_fast("PANIC", nullptr, __FILE__, __LINE__, (m)); \
    } while (0)

// Unreachable
#if defined(__GNUC__) || defined(__clang__)
#define UNREACHABLE()                                                              \
    do {                                                                           \
        ::pts::diagnostics::fail_fast("UNREACHABLE", nullptr, __FILE__, __LINE__); \
        __builtin_unreachable();                                                   \
    } while (0)
#elif defined(_MSC_VER)
#define UNREACHABLE()                                                              \
    do {                                                                           \
        ::pts::diagnostics::fail_fast("UNREACHABLE", nullptr, __FILE__, __LINE__); \
        __assume(0);                                                               \
    } while (0)
#else
#define UNREACHABLE()                                                              \
    do {                                                                           \
        ::pts::diagnostics::fail_fast("UNREACHABLE", nullptr, __FILE__, __LINE__); \
    } while (0)
#endif
