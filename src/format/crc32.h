#pragma once

/// @file crc32.h
/// @brief CRC32C (Castagnoli) checksum with hardware acceleration.
///
/// Uses SSE4.2 _mm_crc32_u64 on x86-64, __crc32cd on AArch64.
/// Software fallback uses the CRC32C polynomial (0x82F63B78 reflected).
/// All paths produce identical results for cross-platform file integrity.

#include <cstddef>
#include <cstdint>

namespace nos {

/// Compute CRC32C checksum with hardware acceleration when available.
///
/// @param data  Pointer to data buffer
/// @param len   Length in bytes
/// @param crc   Initial CRC value (default 0 for new computation)
/// @return CRC32C checksum
uint32_t crc32c(const void* data, size_t len, uint32_t crc = 0);

/// Compute CRC32C checksum using software fallback (portable).
///
/// Always correct, approximately 30x slower than hardware path.
/// Used automatically when SSE4.2 / ARM CRC extensions are unavailable.
///
/// @param data  Pointer to data buffer
/// @param len   Length in bytes
/// @param crc   Initial CRC value (default 0 for new computation)
/// @return CRC32C checksum
uint32_t crc32c_sw(const void* data, size_t len, uint32_t crc = 0);

}  // namespace nos
