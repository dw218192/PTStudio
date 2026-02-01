#pragma once
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/slot/counter.hpp>
#include <boost/scope/scope_exit.hpp>
#include <boost/scope/scope_fail.hpp>
#include <boost/scope/scope_success.hpp>

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

#define _SCOPE_UNIQUE(prefix) BOOST_PP_CAT(prefix, BOOST_PP_SLOT_COUNTER)

#define SCOPE_EXIT auto _SCOPE_UNIQUE(_scope_exit_) = ::pts::detail::exit + [&]() noexcept -> void

#define SCOPE_FAIL auto _SCOPE_UNIQUE(_scope_fail_) = ::pts::detail::fail + [&]() noexcept -> void

#define SCOPE_SUCCESS \
    auto _SCOPE_UNIQUE(_scope_success_) = ::pts::detail::success + [&]() noexcept -> void
