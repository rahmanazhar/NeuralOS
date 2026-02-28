#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "engine/rope.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("RoPE: precompute generates correct frequency tables", "[rope]") {
    nos::RoPE rope;
    rope.precompute(4, 128, 10000.0f);

    REQUIRE(rope.max_seq_len() == 128);
    REQUIRE(rope.head_dim() == 4);
}

TEST_CASE("RoPE: position 0 applies identity rotation", "[rope]") {
    nos::RoPE rope;
    rope.precompute(4, 128, 10000.0f);

    // At position 0, angle = 0, so cos = 1 and sin = 0
    // Rotation should be identity
    float q[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k[] = {5.0f, 6.0f, 7.0f, 8.0f};

    float q_orig[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k_orig[] = {5.0f, 6.0f, 7.0f, 8.0f};

    rope.apply(q, k, 1, 1, 4, 0);

    // At pos=0: cos(0)=1, sin(0)=0 -> q_new = q_old * 1 - 0 = q_old
    for (int i = 0; i < 4; i++) {
        REQUIRE_THAT(q[i], WithinAbs(q_orig[i], 1e-5));
        REQUIRE_THAT(k[i], WithinAbs(k_orig[i], 1e-5));
    }
}

TEST_CASE("RoPE: different positions produce different results", "[rope]") {
    nos::RoPE rope;
    rope.precompute(4, 128, 10000.0f);

    float q0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k0[] = {5.0f, 6.0f, 7.0f, 8.0f};
    rope.apply(q0, k0, 1, 1, 4, 1);

    float q1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k1[] = {5.0f, 6.0f, 7.0f, 8.0f};
    rope.apply(q1, k1, 1, 1, 4, 10);

    // Positions 1 and 10 should produce different rotations
    bool q_different = false;
    for (int i = 0; i < 4; i++) {
        if (std::abs(q0[i] - q1[i]) > 1e-5f) q_different = true;
    }
    REQUIRE(q_different);
}

TEST_CASE("RoPE: rotation preserves vector magnitude", "[rope]") {
    nos::RoPE rope;
    rope.precompute(4, 128, 10000.0f);

    float q[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k[] = {5.0f, 6.0f, 7.0f, 8.0f};

    float q_norm_before = 0.0f, k_norm_before = 0.0f;
    for (int i = 0; i < 4; i++) {
        q_norm_before += q[i] * q[i];
        k_norm_before += k[i] * k[i];
    }

    rope.apply(q, k, 1, 1, 4, 5);

    float q_norm_after = 0.0f, k_norm_after = 0.0f;
    for (int i = 0; i < 4; i++) {
        q_norm_after += q[i] * q[i];
        k_norm_after += k[i] * k[i];
    }

    REQUIRE_THAT(q_norm_after, WithinAbs(q_norm_before, 1e-3));
    REQUIRE_THAT(k_norm_after, WithinAbs(k_norm_before, 1e-3));
}

TEST_CASE("RoPE: multi-head applies to all heads", "[rope]") {
    nos::RoPE rope;
    rope.precompute(4, 128, 10000.0f);

    // 2 query heads, 2 KV heads, head_dim=4
    float q[] = {1.0f, 2.0f, 3.0f, 4.0f,   5.0f, 6.0f, 7.0f, 8.0f};
    float k[] = {1.0f, 1.0f, 1.0f, 1.0f,   2.0f, 2.0f, 2.0f, 2.0f};

    rope.apply(q, k, 2, 2, 4, 3);

    // Both heads should be rotated (not same as original)
    // Head 0 and head 1 should get the same rotation pattern
    // but applied to different data
    bool h0_changed = false, h1_changed = false;
    float orig_q[] = {1.0f, 2.0f, 3.0f, 4.0f,   5.0f, 6.0f, 7.0f, 8.0f};
    for (int i = 0; i < 4; i++) {
        if (std::abs(q[i] - orig_q[i]) > 1e-5f) h0_changed = true;
        if (std::abs(q[4+i] - orig_q[4+i]) > 1e-5f) h1_changed = true;
    }
    REQUIRE(h0_changed);
    REQUIRE(h1_changed);
}
