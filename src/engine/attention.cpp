#include "engine/attention.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace nos {

void Attention::init(int n_heads, int n_kv_heads, int head_dim, int max_seq_len) {
    n_heads_ = n_heads;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    kv_groups_ = n_heads / n_kv_heads;

    score_buf_.resize(static_cast<size_t>(max_seq_len));
}

size_t Attention::kv_cache_per_layer_bytes() const {
    return static_cast<size_t>(2) * static_cast<size_t>(n_kv_heads_)
         * static_cast<size_t>(max_seq_len_) * static_cast<size_t>(head_dim_) * sizeof(float);
}

void Attention::forward(float* output, const float* q, const float* k, const float* v,
                        int pos, float* kv_cache_base) {
    auto sz = [](int x) -> size_t { return static_cast<size_t>(x); };

    size_t kv_head_stride = sz(max_seq_len_) * sz(head_dim_);
    size_t kv_region_size = sz(n_kv_heads_) * kv_head_stride;

    float* k_cache = kv_cache_base;
    float* v_cache = kv_cache_base + kv_region_size;

    // Store current K, V into cache at position pos
    for (int kh = 0; kh < n_kv_heads_; kh++) {
        float* k_dst = k_cache + sz(kh) * kv_head_stride + sz(pos) * sz(head_dim_);
        const float* k_src = k + sz(kh) * sz(head_dim_);
        std::memcpy(k_dst, k_src, sz(head_dim_) * sizeof(float));

        float* v_dst = v_cache + sz(kh) * kv_head_stride + sz(pos) * sz(head_dim_);
        const float* v_src = v + sz(kh) * sz(head_dim_);
        std::memcpy(v_dst, v_src, sz(head_dim_) * sizeof(float));
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int qh = 0; qh < n_heads_; qh++) {
        const float* q_head = q + sz(qh) * sz(head_dim_);
        int kv_head = qh / kv_groups_;

        const float* k_head_cache = k_cache + sz(kv_head) * kv_head_stride;
        const float* v_head_cache = v_cache + sz(kv_head) * kv_head_stride;

        float max_score = -1e30f;
        for (int t = 0; t <= pos; t++) {
            const float* k_t = k_head_cache + sz(t) * sz(head_dim_);
            float dot = 0.0f;
            for (int d = 0; d < head_dim_; d++) {
                dot += q_head[d] * k_t[d];
            }
            score_buf_[sz(t)] = dot * scale;
            if (score_buf_[sz(t)] > max_score) max_score = score_buf_[sz(t)];
        }

        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            score_buf_[sz(t)] = std::exp(score_buf_[sz(t)] - max_score);
            sum_exp += score_buf_[sz(t)];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t <= pos; t++) {
            score_buf_[sz(t)] *= inv_sum;
        }

        float* out_head = output + sz(qh) * sz(head_dim_);
        std::memset(out_head, 0, sz(head_dim_) * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float* v_t = v_head_cache + sz(t) * sz(head_dim_);
            float w = score_buf_[sz(t)];
            for (int d = 0; d < head_dim_; d++) {
                out_head[d] += w * v_t[d];
            }
        }
    }
}

}  // namespace nos
