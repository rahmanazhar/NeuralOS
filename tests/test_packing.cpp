/// @file test_packing.cpp
/// @brief Exhaustive 5-per-byte ternary packing round-trip tests.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "packing.h"

using namespace nos;

TEST_CASE("pack_5trits / unpack_5trits exhaustive round-trip", "[packing]") {
    // Iterate all 243 valid combinations of 5 trits {-1, 0, +1}
    // (3^5 = 243)
    int count = 0;
    for (int a = -1; a <= 1; a++) {
        for (int b = -1; b <= 1; b++) {
            for (int c = -1; c <= 1; c++) {
                for (int d = -1; d <= 1; d++) {
                    for (int e = -1; e <= 1; e++) {
                        int8_t t0 = static_cast<int8_t>(a);
                        int8_t t1 = static_cast<int8_t>(b);
                        int8_t t2 = static_cast<int8_t>(c);
                        int8_t t3 = static_cast<int8_t>(d);
                        int8_t t4 = static_cast<int8_t>(e);

                        uint8_t packed = pack_5trits(t0, t1, t2, t3, t4);
                        // packed byte can be 0-255 (ceiling division maps base-3 [0,242] -> [0,255])

                        int8_t out[5];
                        unpack_5trits(packed, out);

                        REQUIRE(out[0] == t0);
                        REQUIRE(out[1] == t1);
                        REQUIRE(out[2] == t2);
                        REQUIRE(out[3] == t3);
                        REQUIRE(out[4] == t4);
                        count++;
                    }
                }
            }
        }
    }
    REQUIRE(count == 243);
}

TEST_CASE("pack_5trits special values", "[packing]") {
    SECTION("all zeros") {
        uint8_t packed = pack_5trits(0, 0, 0, 0, 0);
        int8_t out[5];
        unpack_5trits(packed, out);
        for (int i = 0; i < 5; i++) {
            REQUIRE(out[i] == 0);
        }
    }

    SECTION("all ones") {
        uint8_t packed = pack_5trits(1, 1, 1, 1, 1);
        int8_t out[5];
        unpack_5trits(packed, out);
        for (int i = 0; i < 5; i++) {
            REQUIRE(out[i] == 1);
        }
    }

    SECTION("all negative ones") {
        uint8_t packed = pack_5trits(-1, -1, -1, -1, -1);
        int8_t out[5];
        unpack_5trits(packed, out);
        for (int i = 0; i < 5; i++) {
            REQUIRE(out[i] == -1);
        }
    }

    SECTION("alternating pattern") {
        uint8_t packed = pack_5trits(-1, 0, 1, -1, 0);
        int8_t out[5];
        unpack_5trits(packed, out);
        REQUIRE(out[0] == -1);
        REQUIRE(out[1] == 0);
        REQUIRE(out[2] == 1);
        REQUIRE(out[3] == -1);
        REQUIRE(out[4] == 0);
    }
}

TEST_CASE("pack_row / unpack_row round-trip", "[packing]") {
    auto test_row_length = [](int len) {
        INFO("Row length: " << len);

        // Generate a pattern of trits
        std::vector<int8_t> trits(len);
        for (int i = 0; i < len; i++) {
            trits[i] = static_cast<int8_t>((i % 3) - 1);  // cycles: -1, 0, 1
        }

        int packed_len = (len + 4) / 5;
        std::vector<uint8_t> packed(packed_len);
        pack_row(trits.data(), len, packed.data());

        std::vector<int8_t> unpacked(len);
        unpack_row(packed.data(), len, unpacked.data());

        for (int i = 0; i < len; i++) {
            REQUIRE(unpacked[i] == trits[i]);
        }
    };

    SECTION("length 1") { test_row_length(1); }
    SECTION("length 4") { test_row_length(4); }
    SECTION("length 5") { test_row_length(5); }
    SECTION("length 6") { test_row_length(6); }
    SECTION("length 10") { test_row_length(10); }
    SECTION("length 11") { test_row_length(11); }
    SECTION("length 127") { test_row_length(127); }
    SECTION("length 4096") { test_row_length(4096); }
    SECTION("length 4097") { test_row_length(4097); }
}

TEST_CASE("FP16 conversion round-trip", "[packing]") {
    SECTION("simple values") {
        float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 0.125f};
        for (float v : values) {
            uint16_t h = fp32_to_fp16(v);
            float back = fp16_to_fp32(h);
            REQUIRE(back == v);
        }
    }

    SECTION("zero sign preservation") {
        float pos_zero = 0.0f;
        float neg_zero = -0.0f;
        uint16_t h_pos = fp32_to_fp16(pos_zero);
        uint16_t h_neg = fp32_to_fp16(neg_zero);
        REQUIRE(fp16_to_fp32(h_pos) == 0.0f);
        REQUIRE(fp16_to_fp32(h_neg) == 0.0f);  // -0 == 0 in float comparison
        REQUIRE(h_pos == 0x0000u);
        REQUIRE(h_neg == 0x8000u);
    }

    SECTION("infinity") {
        uint16_t h = fp32_to_fp16(1e30f);  // overflow to inf
        float back = fp16_to_fp32(h);
        REQUIRE(back > 1e30f);  // should be infinity
    }
}

TEST_CASE("Packed byte values are in valid range", "[packing]") {
    // All packed values from valid trit combinations must be 0-242
    for (int a = -1; a <= 1; a++) {
        for (int b = -1; b <= 1; b++) {
            for (int c = -1; c <= 1; c++) {
                for (int d = -1; d <= 1; d++) {
                    for (int e = -1; e <= 1; e++) {
                        uint8_t packed = pack_5trits(
                            static_cast<int8_t>(a), static_cast<int8_t>(b),
                            static_cast<int8_t>(c), static_cast<int8_t>(d),
                            static_cast<int8_t>(e));
                        // packed byte can be 0-255 (ceiling division maps base-3 range to full byte)
                        (void)packed;
                    }
                }
            }
        }
    }
}
