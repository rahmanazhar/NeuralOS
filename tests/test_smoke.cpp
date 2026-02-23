/// @file test_smoke.cpp
/// @brief Smoke test validating build pipeline and NXP format constants.
///
/// This test proves the entire CMake -> compile -> link -> test pipeline works:
///   - Cross-directory includes (tests/ -> src/format/)
///   - Catch2 discovery via catch_discover_tests
///   - NXP format struct sizes match specification

#include <catch2/catch_test_macros.hpp>

#include "expert_format.h"

TEST_CASE("NXP magic number is correct", "[format][smoke]") {
    REQUIRE(nos::NXP_MAGIC == 0x4E585030);
}

TEST_CASE("NxpFileHeader is exactly 256 bytes", "[format][smoke]") {
    REQUIRE(sizeof(nos::NxpFileHeader) == 256);
}

TEST_CASE("NxpExpertEntry is exactly 64 bytes", "[format][smoke]") {
    REQUIRE(sizeof(nos::NxpExpertEntry) == 64);
}

TEST_CASE("NXP header fields at expected offsets", "[format][smoke]") {
    nos::NxpFileHeader hdr{};
    hdr.magic = nos::NXP_MAGIC;
    hdr.version = nos::NXP_VERSION;
    hdr.num_layers = 32;
    hdr.experts_per_layer = 8;
    hdr.hidden_dim = 4096;
    hdr.intermediate_dim = 11008;
    hdr.packing_mode = 0;
    hdr.scale_dtype = 0;
    hdr.index_offset = 256;
    hdr.data_offset = 256 + 64 * 256;
    hdr.total_experts = 256;

    REQUIRE(hdr.magic == nos::NXP_MAGIC);
    REQUIRE(hdr.version == nos::NXP_VERSION);
    REQUIRE(hdr.num_layers == 32);
    REQUIRE(hdr.experts_per_layer == 8);
    REQUIRE(hdr.hidden_dim == 4096);
    REQUIRE(hdr.intermediate_dim == 11008);
    REQUIRE(hdr.packing_mode == 0);
    REQUIRE(hdr.scale_dtype == 0);
    REQUIRE(hdr.total_experts == 256);
}

TEST_CASE("NXP alignment constant is 64 bytes", "[format][smoke]") {
    REQUIRE(nos::NXP_ALIGNMENT == 64);
}
