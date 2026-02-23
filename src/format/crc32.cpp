/// @file crc32.cpp
/// @brief CRC32C (Castagnoli) implementation with hardware acceleration.
///
/// Uses SSE4.2 on x86-64, ARM CRC extension on AArch64, and a portable
/// software fallback. All paths produce identical results for the CRC32C
/// polynomial (0x1EDC6F41 / reflected 0x82F63B78).

#include "crc32.h"

#include <array>
#include <cstring>

namespace nos {

// ── Software CRC32C Lookup Table ──────────────────────────────────────────────
// CRC32C polynomial: 0x82F63B78 (reflected form of Castagnoli 0x1EDC6F41)
// This is NOT the Ethernet CRC32 polynomial (0xEDB88320).

static constexpr std::array<uint32_t, 256> make_crc32c_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0x82F63B78u;
            } else {
                crc >>= 1;
            }
        }
        table[i] = crc;
    }
    return table;
}

static constexpr auto kCrc32cTable = make_crc32c_table();

uint32_t crc32c_sw(const void* data, size_t len, uint32_t crc) {
    auto p = static_cast<const uint8_t*>(data);
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = kCrc32cTable[static_cast<uint8_t>(crc ^ p[i])] ^ (crc >> 8);
    }
    return ~crc;
}

// ── Hardware-Accelerated CRC32C ───────────────────────────────────────────────

#if defined(__x86_64__) && defined(__SSE4_2__)
#include <nmmintrin.h>

uint32_t crc32c(const void* data, size_t len, uint32_t crc) {
    auto p = static_cast<const uint8_t*>(data);
    crc = ~crc;

    // Process 8 bytes at a time using SSE4.2 CRC32C instruction
    while (len >= 8) {
        uint64_t val;
        std::memcpy(&val, p, sizeof(val));
        crc = static_cast<uint32_t>(_mm_crc32_u64(crc, val));
        p += 8;
        len -= 8;
    }

    // Process remaining bytes one at a time
    while (len > 0) {
        crc = _mm_crc32_u8(crc, *p);
        p++;
        len--;
    }

    return ~crc;
}

#elif defined(__aarch64__)
#include <arm_acle.h>

uint32_t crc32c(const void* data, size_t len, uint32_t crc) {
    auto p = static_cast<const uint8_t*>(data);
    crc = ~crc;

    // Process 8 bytes at a time using ARM CRC extension
    while (len >= 8) {
        uint64_t val;
        std::memcpy(&val, p, sizeof(val));
        crc = __crc32cd(crc, val);
        p += 8;
        len -= 8;
    }

    // Process remaining bytes one at a time
    while (len > 0) {
        crc = __crc32cb(crc, *p);
        p++;
        len--;
    }

    return ~crc;
}

#else

// No hardware acceleration available -- use software fallback
uint32_t crc32c(const void* data, size_t len, uint32_t crc) {
    return crc32c_sw(data, len, crc);
}

#endif

}  // namespace nos
