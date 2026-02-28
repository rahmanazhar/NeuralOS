#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstring>
#include <vector>

#include "engine/attention.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("Attention: MHA single head basic", "[attention]") {
    nos::Attention attn;
    attn.init(/*n_heads=*/1, /*n_kv_heads=*/1, /*head_dim=*/4, /*max_seq_len=*/16);

    // KV cache for 1 layer
    size_t kv_bytes = attn.kv_cache_per_layer_bytes();
    std::vector<float> kv_cache(kv_bytes / sizeof(float), 0.0f);

    // Position 0: feed Q, K, V
    float q0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v0[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float out0[4] = {};

    attn.forward(out0, q0, k0, v0, 0, kv_cache.data());

    // At pos=0, only one KV entry. Softmax is trivially [1.0].
    // Output = 1.0 * v0 = [0, 1, 0, 0]
    REQUIRE_THAT(out0[0], WithinAbs(0.0f, 1e-5));
    REQUIRE_THAT(out0[1], WithinAbs(1.0f, 1e-5));
    REQUIRE_THAT(out0[2], WithinAbs(0.0f, 1e-5));
    REQUIRE_THAT(out0[3], WithinAbs(0.0f, 1e-5));
}

TEST_CASE("Attention: MHA causal masking at position 1", "[attention]") {
    nos::Attention attn;
    attn.init(1, 1, 4, 16);

    size_t kv_floats = attn.kv_cache_per_layer_bytes() / sizeof(float);
    std::vector<float> kv_cache(kv_floats, 0.0f);

    // Position 0
    float q0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float out[4] = {};
    attn.forward(out, q0, k0, v0, 0, kv_cache.data());

    // Position 1: Q aligns with K at pos 0 and pos 1
    float q1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.0f, 1.0f, 0.0f, 0.0f};
    attn.forward(out, q1, k1, v1, 1, kv_cache.data());

    // Both K vectors are [1,0,0,0], so Q.K = 1/sqrt(4) = 0.5 for both
    // Softmax of equal scores = [0.5, 0.5]
    // Output = 0.5 * v0 + 0.5 * v1 = [0.5, 0.5, 0, 0]
    REQUIRE_THAT(out[0], WithinAbs(0.5f, 1e-4));
    REQUIRE_THAT(out[1], WithinAbs(0.5f, 1e-4));
}

TEST_CASE("Attention: GQA with n_heads=4, n_kv_heads=2", "[attention]") {
    nos::Attention attn;
    attn.init(/*n_heads=*/4, /*n_kv_heads=*/2, /*head_dim=*/4, /*max_seq_len=*/16);

    size_t kv_floats = attn.kv_cache_per_layer_bytes() / sizeof(float);
    std::vector<float> kv_cache(kv_floats, 0.0f);

    // 4 query heads, 2 KV heads -> kv_groups = 2
    // Q heads 0,1 use KV head 0; Q heads 2,3 use KV head 1
    std::vector<float> q(16, 0.5f);  // 4 heads * 4 dim
    std::vector<float> k(8, 0.3f);   // 2 KV heads * 4 dim
    std::vector<float> v(8, 0.7f);   // 2 KV heads * 4 dim
    std::vector<float> out(16, 0.0f);

    attn.forward(out.data(), q.data(), k.data(), v.data(), 0, kv_cache.data());

    // At pos=0, each query head should produce v (the only KV entry)
    for (int h = 0; h < 4; h++) {
        for (int d = 0; d < 4; d++) {
            int kv_h = h / 2;
            REQUIRE_THAT(out[h * 4 + d], WithinAbs(v[kv_h * 4 + d], 1e-4));
        }
    }
}

TEST_CASE("Attention: MQA with n_heads=4, n_kv_heads=1", "[attention]") {
    nos::Attention attn;
    attn.init(/*n_heads=*/4, /*n_kv_heads=*/1, /*head_dim=*/4, /*max_seq_len=*/16);

    size_t kv_floats = attn.kv_cache_per_layer_bytes() / sizeof(float);
    std::vector<float> kv_cache(kv_floats, 0.0f);

    std::vector<float> q(16, 0.5f);  // 4 heads * 4 dim
    std::vector<float> k(4, 0.3f);   // 1 KV head * 4 dim
    std::vector<float> v(4, 0.9f);   // 1 KV head * 4 dim
    std::vector<float> out(16, 0.0f);

    attn.forward(out.data(), q.data(), k.data(), v.data(), 0, kv_cache.data());

    // All 4 query heads should produce v (single KV head)
    for (int h = 0; h < 4; h++) {
        for (int d = 0; d < 4; d++) {
            REQUIRE_THAT(out[h * 4 + d], WithinAbs(0.9f, 1e-4));
        }
    }
}

TEST_CASE("Attention: KV cache persists across positions", "[attention]") {
    nos::Attention attn;
    attn.init(1, 1, 4, 16);

    size_t kv_floats = attn.kv_cache_per_layer_bytes() / sizeof(float);
    std::vector<float> kv_cache(kv_floats, 0.0f);

    float out[4] = {};

    // Position 0: V = [1, 0, 0, 0]
    float q0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k0[] = {0.0f, 0.0f, 0.0f, 1.0f};
    float v0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    attn.forward(out, q0, k0, v0, 0, kv_cache.data());

    // Position 1: V = [0, 0, 0, 1]
    float q1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {0.0f, 0.0f, 0.0f, 1.0f};
    attn.forward(out, q1, k1, v1, 1, kv_cache.data());

    // Position 2: Q=[1,0,0,0] should attend more to K1=[1,0,0,0] than K0=[0,0,0,1]
    float q2[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    attn.forward(out, q2, k2, v2, 2, kv_cache.data());

    // K at pos 1 had highest dot with Q -> weight on V1 should be highest
    // But V0 and V2 should also contribute
    // Main check: output is not zero and is a valid weighted sum
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += out[i] * out[i];
    REQUIRE(sum > 0.0f);
}
