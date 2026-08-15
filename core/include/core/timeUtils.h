#pragma once

#include <chrono>

namespace pts {

/// Calculates the time elapsed since a given start time point.
/// @param start The starting time point.
/// @return The elapsed time in seconds as a double.
auto time_since_start(const std::chrono::steady_clock::time_point& start) -> double;

}  // namespace pts
