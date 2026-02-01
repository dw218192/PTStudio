#include <core/timeUtils.h>

namespace pts {

auto time_since_start(const std::chrono::steady_clock::time_point& start) -> double {
    auto const now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

}  // namespace pts
