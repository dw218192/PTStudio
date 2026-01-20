#pragma once
#include <cstdint>

/**
 * FNV-1a 64-bit hash function. Used to generate a unique identifier for a plugin interface.
 */
constexpr uint64_t fnv1a64(const char* s) {
    uint64_t h = 14695981039346656037ull;
    for (; *s; ++s) {
        h ^= static_cast<uint8_t>(*s);
        h *= 1099511628211ull;
    }
    return h;
}
