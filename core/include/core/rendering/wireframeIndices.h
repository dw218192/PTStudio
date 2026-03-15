#pragma once

#include <cstdint>
#include <vector>

namespace pts::rendering {

/// Expand triangle indices (a,b,c,...) to line-list indices (a,b, b,c, c,a,...).
/// @param tri_indices Triangle index data (size must be a multiple of 3).
/// @return Line-list indices with 2x the number of entries (6 per triangle).
std::vector<uint32_t> expand_wireframe_indices(const std::vector<uint32_t>& tri_indices);

}  // namespace pts::rendering
