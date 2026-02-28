#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "engine/rmsnorm.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("rmsnorm: known input produces correct output", "[rmsnorm]") {
    // Input: [1.0, 2.0, 3.0, 4.0], weights all 1.0, eps=1e-5
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {};

    nos::rmsnorm(out, x, w, 4, 1e-5f);

    // Manual: ss = 1+4+9+16 = 30, rms = 1/sqrt(30/4 + 1e-5) = 1/sqrt(7.50001)
    float rms = 1.0f / std::sqrt(30.0f / 4.0f + 1e-5f);
    REQUIRE_THAT(out[0], WithinAbs(1.0f * rms, 1e-5));
    REQUIRE_THAT(out[1], WithinAbs(2.0f * rms, 1e-5));
    REQUIRE_THAT(out[2], WithinAbs(3.0f * rms, 1e-5));
    REQUIRE_THAT(out[3], WithinAbs(4.0f * rms, 1e-5));
}

TEST_CASE("rmsnorm: weights scale output", "[rmsnorm]") {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {2.0f, 0.5f, 1.0f, 3.0f};
    float out[4] = {};

    nos::rmsnorm(out, x, w, 4, 1e-5f);

    float rms = 1.0f / std::sqrt(30.0f / 4.0f + 1e-5f);
    REQUIRE_THAT(out[0], WithinAbs(1.0f * rms * 2.0f, 1e-5));
    REQUIRE_THAT(out[1], WithinAbs(2.0f * rms * 0.5f, 1e-5));
    REQUIRE_THAT(out[2], WithinAbs(3.0f * rms * 1.0f, 1e-5));
    REQUIRE_THAT(out[3], WithinAbs(4.0f * rms * 3.0f, 1e-5));
}

TEST_CASE("rmsnorm: zero input produces zero output", "[rmsnorm]") {
    float x[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {};

    nos::rmsnorm(out, x, w, 4, 1e-5f);

    for (int i = 0; i < 4; i++) {
        REQUIRE_THAT(out[i], WithinAbs(0.0f, 1e-5));
    }
}

TEST_CASE("rmsnorm: large values don't overflow", "[rmsnorm]") {
    float x[] = {1000.0f, 2000.0f, 3000.0f, 4000.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {};

    nos::rmsnorm(out, x, w, 4, 1e-5f);

    // Output should have unit RMS (since weights are 1)
    float ss = 0.0f;
    for (int i = 0; i < 4; i++) ss += out[i] * out[i];
    float output_rms = std::sqrt(ss / 4.0f);
    REQUIRE_THAT(output_rms, WithinAbs(1.0f, 1e-3));
}
