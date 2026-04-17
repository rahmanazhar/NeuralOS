#pragma once

/// @file packing.h
/// @brief 5-per-byte ternary encode/decode utilities and FP16 conversion helpers.
///
/// Encodes 5 ternary values {-1, 0, +1} into a single byte (0-242 range).
/// Uses multiplication-based decode (no division/modulo) for SIMD friendliness.
/// Source: compilade.net/blog/ternary-packing

#include <cstddef>
#include <cstdint>

namespace nos {

// ── FP16 Conversion Helpers ──────────────────────────────────────────────────
// On platforms without native _Float16 support, we store FP16 as uint16_t
// and convert manually via bit manipulation.

/// Convert a half-precision float (stored as uint16_t) to single-precision float.
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x03FFu;

    if (exponent == 0) {
        if (mantissa == 0) {
            // +/- zero
            uint32_t result = sign;
            float f;
            __builtin_memcpy(&f, &result, sizeof(f));
            return f;
        }
        // Subnormal: normalize
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= ~0x0400u;
    } else if (exponent == 31) {
        // Inf or NaN
        uint32_t result = sign | 0x7F800000u | (mantissa << 13);
        float f;
        __builtin_memcpy(&f, &result, sizeof(f));
        return f;
    }

    uint32_t result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    float f;
    __builtin_memcpy(&f, &result, sizeof(f));
    return f;
}

/// Convert a single-precision float to half-precision float (stored as uint16_t).
/// Uses round-to-nearest (ties round up) for minimal conversion error.
inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));

    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    uint32_t mantissa = bits & 0x007FFFFFu;

    if (exponent > 15) {
        // Overflow -> Inf
        return sign | 0x7C00u;
    }
    if (exponent < -14) {
        // Underflow -> zero (subnormals not worth handling for scale factors)
        return sign;
    }

    // Round-to-nearest: add bit 12 (0.5 ULP of FP16 mantissa)
    mantissa += 0x00001000u;
    if (mantissa & 0x00800000u) {
        // Carry overflowed into hidden bit — increment exponent, reset mantissa
        mantissa = 0;
        exponent++;
        if (exponent > 15) {
            return sign | 0x7C00u;  // overflow to Inf
        }
    }

    return sign | static_cast<uint16_t>((exponent + 15) << 10)
         | static_cast<uint16_t>(mantissa >> 13);
}

// ── 5-per-byte Ternary Packing ──────────────────────────────────────────────

/// Encode 5 trits {-1, 0, +1} into a single byte (0-242 range).
///
/// Maps each trit from {-1, 0, +1} to {0, 1, 2}, then encodes as base-3:
///   value = (t0+1)*81 + (t1+1)*27 + (t2+1)*9 + (t3+1)*3 + (t4+1)
/// Then ceiling division to map 0-242 into byte range:
///   byte = (value * 256 + 242) / 243
inline uint8_t pack_5trits(int8_t t0, int8_t t1, int8_t t2,
                            int8_t t3, int8_t t4) {
    uint32_t b = static_cast<uint32_t>(t0 + 1) * 81
               + static_cast<uint32_t>(t1 + 1) * 27
               + static_cast<uint32_t>(t2 + 1) * 9
               + static_cast<uint32_t>(t3 + 1) * 3
               + static_cast<uint32_t>(t4 + 1);
    return static_cast<uint8_t>((b * 256 + 242) / 243);
}

/// Decode a single byte into 5 trits {-1, 0, +1}.
///
/// Uses multiplication-based decode (SIMD-friendly, no division/modulo):
///   m = q * 3; trit = (m >> 8) - 1; q = m & 0xFF
inline void unpack_5trits(uint8_t byte, int8_t out[5]) {
    uint16_t q = byte;
    for (int i = 0; i < 5; i++) {
        uint16_t m = q * 3;
        out[i] = static_cast<int8_t>(m >> 8) - 1;
        q = m & 0xFF;
    }
}

/// Pack a row of trits into packed bytes.
///
/// @param trits  Input trit array {-1, 0, +1}
/// @param count  Number of trits (will be padded with zeros if not divisible by 5)
/// @param packed Output byte array, must be pre-allocated with at least (count + 4) / 5 bytes
inline void pack_row(const int8_t* trits, int count, uint8_t* packed) {
    int num_bytes = (count + 4) / 5;
    for (int b = 0; b < num_bytes; b++) {
        int base = b * 5;
        int8_t t0 = (base + 0 < count) ? trits[base + 0] : 0;
        int8_t t1 = (base + 1 < count) ? trits[base + 1] : 0;
        int8_t t2 = (base + 2 < count) ? trits[base + 2] : 0;
        int8_t t3 = (base + 3 < count) ? trits[base + 3] : 0;
        int8_t t4 = (base + 4 < count) ? trits[base + 4] : 0;
        packed[b] = pack_5trits(t0, t1, t2, t3, t4);
    }
}

/// Unpack packed bytes into a row of trits.
///
/// @param packed  Input packed byte array
/// @param count   Number of trits to produce
/// @param trits   Output trit array, must be pre-allocated with at least count elements
inline void unpack_row(const uint8_t* packed, int count, int8_t* trits) {
    int num_bytes = (count + 4) / 5;
    int8_t tmp[5];
    for (int b = 0; b < num_bytes; b++) {
        unpack_5trits(packed[b], tmp);
        int base = b * 5;
        for (int t = 0; t < 5 && base + t < count; t++) {
            trits[base + t] = tmp[t];
        }
    }
}

}  // namespace nos
