#pragma once
#include <boost/outcome/outcome.hpp>
#include <cstdint>

namespace pts {
enum class ErrorCode : std::int32_t {
    Ok = 0,
    InternalError = 1,
    InvalidArgument = 2,
};
template <typename R, typename E = ErrorCode>
using result = ::BOOST_OUTCOME_V2_NAMESPACE::result<R, E>;
}  // namespace pts