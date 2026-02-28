#include "engine/rope.h"
#include <cmath>

namespace nos {

void RoPE::precompute(int head_dim, int max_seq_len, float theta) {
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    half_dim_ = head_dim / 2;

    auto sz = [](int x) -> size_t { return static_cast<size_t>(x); };

    cos_table_.resize(sz(max_seq_len) * sz(half_dim_));
    sin_table_.resize(sz(max_seq_len) * sz(half_dim_));

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim_; i++) {
            float freq = 1.0f / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;
            cos_table_[sz(pos) * sz(half_dim_) + sz(i)] = std::cos(angle);
            sin_table_[sz(pos) * sz(half_dim_) + sz(i)] = std::sin(angle);
        }
    }
}

void RoPE::apply(float* q, float* k, int n_heads, int n_kv_heads,
                 int head_dim, int pos) const {
    auto sz = [](int x) -> size_t { return static_cast<size_t>(x); };

    const float* cos_row = cos_table_.data() + sz(pos) * sz(half_dim_);
    const float* sin_row = sin_table_.data() + sz(pos) * sz(half_dim_);
    int half = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float* qh = q + sz(h) * sz(head_dim);
        for (int i = 0; i < half; i++) {
            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_row[i] - q1 * sin_row[i];
            qh[i + half]  = q0 * sin_row[i] + q1 * cos_row[i];
        }
    }

    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + sz(h) * sz(head_dim);
        for (int i = 0; i < half; i++) {
            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_row[i] - k1 * sin_row[i];
            kh[i + half]  = k0 * sin_row[i] + k1 * cos_row[i];
        }
    }
}

}  // namespace nos
