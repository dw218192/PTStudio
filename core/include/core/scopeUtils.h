#pragma once
#include <boost/preprocessor/cat.hpp>
#include <boost/scope/scope_exit.hpp>
#include <boost/scope/scope_fail.hpp>
#include <boost/scope/scope_success.hpp>
#include <utility>

namespace pts::detail {
struct exit_tag {
    template <class F>
    auto operator+(F&& f) const {
        return ::boost::scope::scope_exit(std::forward<F>(f));
    }
};
struct fail_tag {
    template <class F>
    auto operator+(F&& f) const {
        return ::boost::scope::scope_fail(std::forward<F>(f));
    }
};
struct success_tag {
    template <class F>
    auto operator+(F&& f) const {
        return ::boost::scope::scope_success(std::forward<F>(f));
    }
};

inline constexpr exit_tag exit{};
inline constexpr fail_tag fail{};
inline constexpr success_tag success{};
}  // namespace pts::detail

#if defined(__COUNTER__)
#define _SCOPE_UNIQUE(prefix) BOOST_PP_CAT(prefix, __COUNTER__)
#else
#define _SCOPE_UNIQUE(prefix) BOOST_PP_CAT(prefix, __LINE__)
#endif

#define SCOPE_EXIT auto _SCOPE_UNIQUE(_scope_exit_) = ::pts::detail::exit + [&]() noexcept -> void

#define SCOPE_FAIL auto _SCOPE_UNIQUE(_scope_fail_) = ::pts::detail::fail + [&]() noexcept -> void

#define SCOPE_SUCCESS \
    auto _SCOPE_UNIQUE(_scope_success_) = ::pts::detail::success + [&]() noexcept -> void
