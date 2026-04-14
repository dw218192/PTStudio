#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace pts::rendering {

inline uint16_t float_to_half(float f) {
    // IEEE 754 float32 -> float16 conversion
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    if (exponent == 0xFF - 127 + 15) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7C00);              // inf
        return static_cast<uint16_t>(sign | 0x7C00 | std::max(mantissa >> 13, 1u));  // nan
    }
    if (exponent > 30) return static_cast<uint16_t>(sign | 0x7C00);  // overflow -> inf

    return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

}  // namespace pts::rendering
