/// @file test_crc32.cpp
/// @brief CRC32C correctness tests across hardware/software paths.
///
/// Validates: known test vectors, hardware vs software equivalence,
/// incremental computation, and unaligned buffer handling.

#include <catch2/catch_test_macros.hpp>

#include "crc32.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

TEST_CASE("CRC32C known test vectors", "[crc32]") {
    SECTION("Empty string") {
        uint32_t crc = nos::crc32c("", 0);
        REQUIRE(crc == 0x00000000);
    }

    SECTION("Standard test vector: '123456789' -> 0xE3069283") {
        const char* data = "123456789";
        uint32_t crc = nos::crc32c(data, 9);
        REQUIRE(crc == 0xE3069283);
    }
}

TEST_CASE("CRC32C software known test vectors", "[crc32]") {
    SECTION("Empty string") {
        uint32_t crc = nos::crc32c_sw("", 0);
        REQUIRE(crc == 0x00000000);
    }

    SECTION("Standard test vector: '123456789' -> 0xE3069283") {
        const char* data = "123456789";
        uint32_t crc = nos::crc32c_sw(data, 9);
        REQUIRE(crc == 0xE3069283);
    }
}

TEST_CASE("CRC32C hardware vs software produce identical results", "[crc32]") {
    // Generate 1MB of random data
    constexpr size_t kSize = 1024 * 1024;
    std::vector<uint8_t> data(kSize);
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::generate(data.begin(), data.end(), [&]() {
        return static_cast<uint8_t>(rng());
    });

    uint32_t hw = nos::crc32c(data.data(), data.size());
    uint32_t sw = nos::crc32c_sw(data.data(), data.size());
    REQUIRE(hw == sw);
}

TEST_CASE("CRC32C incremental computation", "[crc32]") {
    // Compute in one shot
    const char* data = "Hello, World! This is a test of incremental CRC32C.";
    size_t len = std::strlen(data);
    uint32_t one_shot = nos::crc32c(data, len);

    // Compute in 3 chunks -- passing intermediate CRC
    size_t chunk1 = 13;  // "Hello, World!"
    size_t chunk2 = 20;  // " This is a test of "
    size_t chunk3 = len - chunk1 - chunk2;

    uint32_t incremental = nos::crc32c(data, chunk1);
    incremental = nos::crc32c(data + chunk1, chunk2, incremental);
    incremental = nos::crc32c(data + chunk1 + chunk2, chunk3, incremental);

    REQUIRE(one_shot == incremental);
}

TEST_CASE("CRC32C incremental computation (software)", "[crc32]") {
    const char* data = "Hello, World! This is a test of incremental CRC32C.";
    size_t len = std::strlen(data);
    uint32_t one_shot = nos::crc32c_sw(data, len);

    size_t chunk1 = 13;
    size_t chunk2 = 20;
    size_t chunk3 = len - chunk1 - chunk2;

    uint32_t incremental = nos::crc32c_sw(data, chunk1);
    incremental = nos::crc32c_sw(data + chunk1, chunk2, incremental);
    incremental = nos::crc32c_sw(data + chunk1 + chunk2, chunk3, incremental);

    REQUIRE(one_shot == incremental);
}

TEST_CASE("CRC32C on unaligned buffer", "[crc32]") {
    // Allocate buffer with offset to create unaligned access
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> aligned_buf(kSize + 16);
    std::mt19937 rng(123);
    std::generate(aligned_buf.begin(), aligned_buf.end(), [&]() {
        return static_cast<uint8_t>(rng());
    });

    // Compute CRC on aligned data
    uint32_t aligned_crc = nos::crc32c(aligned_buf.data(), kSize);

    // Copy same data to unaligned position (offset by 1 byte)
    std::vector<uint8_t> unaligned_buf(kSize + 16);
    std::memcpy(unaligned_buf.data() + 1, aligned_buf.data(), kSize);

    // Compute CRC on unaligned data -- should produce same result
    uint32_t unaligned_crc = nos::crc32c(unaligned_buf.data() + 1, kSize);

    REQUIRE(aligned_crc == unaligned_crc);
}

TEST_CASE("CRC32C various sizes", "[crc32]") {
    // Test sizes that exercise different code paths in the hardware implementation:
    // 0, 1-7 bytes (tail only), 8 bytes (one u64), 9-15 (one u64 + tail),
    // and larger sizes.

    std::vector<uint8_t> data(128);
    std::mt19937 rng(999);
    std::generate(data.begin(), data.end(), [&]() {
        return static_cast<uint8_t>(rng());
    });

    for (size_t sz : {0, 1, 3, 7, 8, 9, 15, 16, 31, 32, 63, 64, 127, 128}) {
        uint32_t hw = nos::crc32c(data.data(), sz);
        uint32_t sw = nos::crc32c_sw(data.data(), sz);
        REQUIRE(hw == sw);
    }
}
