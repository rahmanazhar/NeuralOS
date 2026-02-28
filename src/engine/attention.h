#pragma once

/// @file attention.h
/// @brief Multi-head attention supporting MHA, GQA, and MQA with KV cache.

#include <cstdint>
#include <vector>

namespace nos {

/// Attention layer supporting MHA/GQA/MQA with KV cache.
///
/// KV cache layout per layer: [2][n_kv_heads][max_seq_len][head_dim]
/// where [2] is for K and V. The caller provides the base pointer
/// for this layer's KV cache region.
class Attention {
public:
    /// Initialize attention parameters.
    ///
    /// @param n_heads      Number of query heads
    /// @param n_kv_heads   Number of KV heads (== n_heads for MHA, < for GQA, == 1 for MQA)
    /// @param head_dim     Dimension per head
    /// @param max_seq_len  Maximum sequence length for KV cache
    void init(int n_heads, int n_kv_heads, int head_dim, int max_seq_len);

    /// Compute attention for a single token at position pos.
    ///
    /// Q, K, V projections are already computed by the caller.
    /// Stores K, V in KV cache at the given position.
    ///
    /// @param output        Output buffer [n_heads * head_dim]
    /// @param q             Query buffer [n_heads * head_dim]
    /// @param k             Key buffer [n_kv_heads * head_dim]
    /// @param v             Value buffer [n_kv_heads * head_dim]
    /// @param pos           Current token position in sequence
    /// @param kv_cache_base Base pointer to this layer's KV cache
    void forward(float* output, const float* q, const float* k, const float* v,
                 int pos, float* kv_cache_base);

    /// Size in bytes of KV cache needed per layer.
    size_t kv_cache_per_layer_bytes() const;

    int n_heads() const { return n_heads_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int head_dim() const { return head_dim_; }

private:
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_seq_len_ = 0;
    int kv_groups_ = 0;  // n_heads / n_kv_heads

    // Workspace for attention scores (allocated once)
    std::vector<float> score_buf_;
};

}  // namespace nos
