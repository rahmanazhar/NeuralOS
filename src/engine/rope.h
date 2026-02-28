#pragma once

/// @file rope.h
/// @brief Rotary Position Embedding (RoPE) for transformer attention.

#include <cstdint>
#include <vector>

namespace nos {

/// Rotary position embedding with precomputed frequency tables.
class RoPE {
public:
    /// Precompute cos/sin tables for all positions.
    ///
    /// @param head_dim     Dimension per attention head
    /// @param max_seq_len  Maximum sequence length
    /// @param theta        Base frequency (default 10000.0)
    void precompute(int head_dim, int max_seq_len, float theta = 10000.0f);

    /// Apply rotary embedding to query and key vectors at a given position.
    ///
    /// @param q          Query buffer [n_heads * head_dim]
    /// @param k          Key buffer [n_kv_heads * head_dim]
    /// @param n_heads    Number of query heads
    /// @param n_kv_heads Number of KV heads
    /// @param head_dim   Dimension per head
    /// @param pos        Token position in sequence
    void apply(float* q, float* k, int n_heads, int n_kv_heads,
               int head_dim, int pos) const;

    int max_seq_len() const { return max_seq_len_; }
    int head_dim() const { return head_dim_; }

private:
    std::vector<float> cos_table_;  // [max_seq_len * half_dim]
    std::vector<float> sin_table_;  // [max_seq_len * half_dim]
    int max_seq_len_ = 0;
    int head_dim_ = 0;
    int half_dim_ = 0;
};

}  // namespace nos
