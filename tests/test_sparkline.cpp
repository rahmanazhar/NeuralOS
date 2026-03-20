/// @file test_sparkline.cpp
/// @brief Catch2 tests for the sparkline renderer.

#include <catch2/catch_test_macros.hpp>

#include "dashboard/sparkline.h"

#include <cstring>
#include <vector>

using namespace nos;

TEST_CASE("All zeros returns flat sparkline", "[sparkline]") {
    float values[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::string result = render_sparkline(values, 5, 5);
    // All equal values -> middle bar (U+2584, level 3)
    // Each character should be the same
    REQUIRE(result.size() == 15);  // 5 chars * 3 bytes each
    // All characters should be identical
    REQUIRE(result.substr(0, 3) == result.substr(3, 3));
    REQUIRE(result.substr(0, 3) == result.substr(6, 3));
}

TEST_CASE("Ascending values produce ascending blocks", "[sparkline]") {
    float values[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::string result = render_sparkline(values, 8, 8);
    REQUIRE(result.size() == 24);  // 8 chars * 3 bytes each

    // First character should be lowest block, last should be highest
    // U+2581 = E2 96 81 (level 0)
    // U+2588 = E2 96 88 (level 7)
    std::string first_char = result.substr(0, 3);
    std::string last_char = result.substr(21, 3);
    REQUIRE(first_char == "\xe2\x96\x81");
    REQUIRE(last_char == "\xe2\x96\x88");

    // Verify ordering: each char should be >= previous
    for (size_t i = 3; i < result.size(); i += 3) {
        unsigned char prev_level = static_cast<unsigned char>(result[i - 1]);
        unsigned char curr_level = static_cast<unsigned char>(result[i + 2]);
        REQUIRE(curr_level >= prev_level);
    }
}

TEST_CASE("Single value returns one character", "[sparkline]") {
    float values[] = {42.0f};
    std::string result = render_sparkline(values, 1, 1);
    REQUIRE(result.size() == 3);  // One UTF-8 character = 3 bytes
}

TEST_CASE("Width parameter controls output length", "[sparkline]") {
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    // Request width=3, should use last 3 values
    std::string result = render_sparkline(values, 5, 3);
    REQUIRE(result.size() == 9);  // 3 chars * 3 bytes

    // Request width=10, should pad with spaces
    result = render_sparkline(values, 5, 10);
    // 5 sparkline chars (15 bytes) + 5 spaces (5 bytes)
    REQUIRE(result.size() == 20);
}

TEST_CASE("Negative values handled as zero", "[sparkline]") {
    float values[] = {-5.0f, -2.0f, 0.0f, 3.0f, 7.0f};
    std::string result = render_sparkline(values, 5, 5);
    // Should not crash, negatives clamped to 0
    REQUIRE(result.size() == 15);  // 5 chars * 3 bytes

    // First two values are clamped to 0, so they should be the lowest bar
    std::string first = result.substr(0, 3);
    std::string second = result.substr(3, 3);
    REQUIRE(first == second);  // Both clamped to 0
    REQUIRE(first == "\xe2\x96\x81");  // Lowest block
}

TEST_CASE("Empty input returns empty string", "[sparkline]") {
    REQUIRE(render_sparkline(nullptr, 0, 5).empty());
    float v = 1.0f;
    REQUIRE(render_sparkline(&v, 0, 5).empty());
    REQUIRE(render_sparkline(nullptr, 5, 5).empty());
}

TEST_CASE("Width zero returns empty string", "[sparkline]") {
    float values[] = {1.0f, 2.0f};
    REQUIRE(render_sparkline(values, 2, 0).empty());
}

TEST_CASE("All equal non-zero values returns flat sparkline", "[sparkline]") {
    float values[] = {5.0f, 5.0f, 5.0f};
    std::string result = render_sparkline(values, 3, 3);
    REQUIRE(result.size() == 9);
    // All same value -> all same character (middle bar)
    REQUIRE(result.substr(0, 3) == result.substr(3, 3));
    REQUIRE(result.substr(0, 3) == result.substr(6, 3));
}
