#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pts::rendering {

/// Expand triangle indices (a,b,c,...) to line-list indices (a,b, b,c, c,a,...).
/// @param tri_indices Pointer to triangle index data.
/// @param tri_index_count Number of triangle indices (must be a multiple of 3).
/// @return Line-list indices with 2x the number of entries (6 per triangle).
std::vector<uint32_t> expand_wireframe_indices(const uint32_t* tri_indices, size_t tri_index_count);

}  // namespace pts::rendering
